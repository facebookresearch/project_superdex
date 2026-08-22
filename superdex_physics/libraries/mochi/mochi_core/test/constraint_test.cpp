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
#include <mochi_core/linear_algebra/krylov_interop.h>
#include <mochi_core/linear_algebra/utils/matrix_conversions.h>
#include <mochi_core/test/fem_rod_test_helpers.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/constraints.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/rodrigues_utils.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

using namespace mochi;

namespace {
Quaternion const kqI = Quaternion::Identity();

constexpr real ToRads(real deg) {
  return kRadiansPerDegree * deg;
}

constexpr auto quatx = [](real rads) { return Quaternion::RotationX(rads); };
constexpr auto quaty = [](real rads) { return Quaternion::RotationY(rads); };
constexpr auto quatz = [](real rads) { return Quaternion::RotationZ(rads); };

constexpr Real3 kv0 = {0_r, 0_r, 0_r};
constexpr Real3 kvx = {1_r, 0_r, 0_r};
constexpr Real3 kvy = {0_r, 1_r, 0_r};
constexpr Real3 kvz = {0_r, 0_r, 1_r};
constexpr Real3 kv123 = kvx + 2_r * kvy + 3_r * kvz;

} // namespace

/// EvalDeformableNodeFixedConstraint ---------------------------------------------
TEST(Constraints, DeformableNodeValue) {
  // Define method to evaluate constraint
  auto testCVal = [](Real3 const& disp,
                     Real3 const& t,
                     Real3 const& r,
                     Quaternion const& q,
                     Real3 const& p,
                     Real3 const& cVal) {
    ColumnVector<real, 3> val;
    TransformRT tx(q, t);
    EvalDeformableNodeFixedConstraint(tx, r + disp, p, &val, nullptr, nullptr);

    real constexpr kTol = 1.0e-3_r;
    for (int i = 0; i < 3; ++i) {
      EXPECT_NEAR_RTOL(cVal[i], val[i], kTol);
    }
  };

  testCVal(kv0, kv0, kv0, kqI, kv0, kv0);
  testCVal(kv123, kv0, kv0, kqI, kv0, kv123);
  testCVal(kv0, kv123, kv0, kqI, kv0, kv123);
  testCVal(kv0, kv0, kv0, quatz(ToRads(90_r)), kv0, kv0);
  testCVal(kv0, kv0, kv0, kqI, kv123, -kv123);
  testCVal(kvx, kv0, kv0, kqI, kv123, -2_r * kvy - 3_r * kvz);
  testCVal(kvx, kv0, kv0, quatz(ToRads(90_r)), kv0, kvy);
  testCVal(kvx, kvx, kv0, quatz(ToRads(90_r)), kvx, kvy);

  auto uFromV = [](Real3 v, // Method to extract displacement from given context
                   Real3 r,
                   Real3 t,
                   Real3 p,
                   Quaternion q) {
    return TransformRT(q.GetConjugate(), -r).TransformPoint(p + v - t);
  };

  std::vector<Real3> vs = {kv0, kvx, kvy, kvz, kv123};
  std::vector<Quaternion> qs = {
      kqI,
      quatx(ToRads(90_r)),
      quatx(ToRads(180_r)),
      quatx(ToRads(-90_r)),
      quatx(ToRads(-360_r)),
      quaty(ToRads(90_r)),
      quaty(ToRads(180_r)),
      quaty(ToRads(-90_r)),
      quaty(ToRads(-360_r)),
      quatz(ToRads(90_r)),
      quatz(ToRads(180_r)),
      quatz(ToRads(-90_r)),
      quatz(ToRads(-360_r)),
  };

  for (auto v : vs) {
    for (auto t : vs) {
      for (auto p : vs) {
        for (auto r : vs) {
          for (auto q : qs) {
            Real3 u = uFromV(v, r, t, p, q);
            testCVal(u, t, r, q, p, v);
          }
        }
      }
    }
  }
}

TEST(Constraints, DeformableNodeJacobian) {
  // Define method to evaluate Jacobian through finite differences
  auto testFD =
      [](Real3 disp, Real3 const& t, Real3 const& r, Quaternion const& q, Real3 const& p) {
        ColumnVector<real, 3> val;
        RowMatrix<real, 3, 3> Jac;
        TransformRT tx(q, t);
        EvalDeformableNodeFixedConstraint(tx, r + disp, p, &val, &Jac, nullptr);

        auto JacFD = Jac.Duplicate();
        real constexpr kDelta = 1.0e-3_r;
        real constexpr kTol = 1.0e-3_r;
        for (int i = 0; i < 3; ++i) {
          real v0 = disp[i];
          real dx = std::max(kDelta, std::abs(kDelta * disp[i]));
          disp[i] += dx;
          EvalDeformableNodeFixedConstraint(tx, r + disp, p, &val, nullptr, nullptr);
          auto valp = val.Duplicate();

          disp[i] -= 2_r * dx;
          EvalDeformableNodeFixedConstraint(tx, r + disp, p, &val, nullptr, nullptr);
          auto valm = val.Duplicate();
          disp[i] = v0;

          JacFD.Col(i) = (valp - valm) * (1_r / (2_r * dx));
        }
        EXPECT_LT((RowMatrix<real, 3, 3>(JacFD - Jac)).Norm(), kTol * JacFD.Norm());
      };

  std::vector<Real3> vs = {kv0, kvx, kvy, kvz, kv123};
  std::vector<Quaternion> qs = {
      kqI,
      quatx(ToRads(90_r)),
      quatx(ToRads(180_r)),
      quatx(ToRads(-90_r)),
      quatx(ToRads(-360_r)),
      quaty(ToRads(90_r)),
      quaty(ToRads(180_r)),
      quaty(ToRads(-90_r)),
      quaty(ToRads(-360_r)),
      quatz(ToRads(90_r)),
      quatz(ToRads(180_r)),
      quatz(ToRads(-90_r)),
      quatz(ToRads(-360_r)),
  };
  for (auto u : vs) {
    for (auto t : vs) {
      for (auto r : vs) {
        for (auto q : qs) {
          for (auto p : vs) {
            testFD(u, t, r, q, p);
          }
        }
      }
    }
  }
}

TEST(Constraints, DeformableNodeJacobianTarget) {
  auto testFD =
      [](Real3 const& disp, Real3 const& t, Real3 const& r, Quaternion const& q, Real3 p) {
        ColumnVector<real, 3> val;
        RowMatrix<real, 3, 3> Jac;
        TransformRT tx(q, t);
        EvalDeformableNodeFixedConstraint(tx, r + disp, p, &val, nullptr, &Jac);

        auto JacFD = Jac.Duplicate();
        real constexpr kDelta = 1.0e-3_r;
        real constexpr kTol = 1.0e-3_r;
        for (int i = 0; i < 3; ++i) {
          real v0 = p[i];
          p[i] += kDelta;
          EvalDeformableNodeFixedConstraint(tx, r + disp, p, &val, nullptr, nullptr);
          auto valp = val.Duplicate();

          p[i] -= 2_r * kDelta;
          EvalDeformableNodeFixedConstraint(tx, r + disp, p, &val, nullptr, nullptr);
          auto valm = val.Duplicate();
          p[i] = v0;

          JacFD.Col(i) = (valp - valm) * (1_r / (2_r * kDelta));
        }
        EXPECT_LT((RowMatrix<real, 3, 3>(JacFD - Jac)).Norm(), kTol * JacFD.Norm());
      };

  std::vector<Real3> vs = {kv0, kvx, kvy, kvz, kv123};
  std::vector<Quaternion> qs = {
      kqI,
      quatx(ToRads(90_r)),
      quatx(ToRads(180_r)),
      quatx(ToRads(-90_r)),
      quatx(ToRads(-360_r)),
      quaty(ToRads(90_r)),
      quaty(ToRads(180_r)),
      quaty(ToRads(-90_r)),
      quaty(ToRads(-360_r)),
      quatz(ToRads(90_r)),
      quatz(ToRads(180_r)),
      quatz(ToRads(-90_r)),
      quatz(ToRads(-360_r)),
  };
  for (auto u : vs) {
    for (auto t : vs) {
      for (auto r : vs) {
        for (auto q : qs) {
          for (auto p : vs) {
            testFD(u, t, r, q, p);
          }
        }
      }
    }
  }
}

/// EvalRotationFixedConstraint ---------------------------------------------
TEST(Constraints, RotationValue) {
  // Define method to evaluate constraint value
  auto testCVal = [](Quaternion const& qs, Quaternion const& qt, Quaternion const& ql, Real3 cVal) {
    ColumnVector<real, 3> val;
    real constexpr kTol = 1.0e-2_r;

    EvalRotationFixedConstraint(qs, ql, qt, &val, nullptr, nullptr);

    for (int i = 0; i < 3; ++i) {
      EXPECT_NEAR_RTOL(cVal[i], val[i], kTol);
    }
  };

  std::vector<Real2> angles = {
      {ToRads(0_r), ToRads(0_r)},
      {ToRads(15_r), ToRads(15_r)},
      {ToRads(90_r), ToRads(90_r)},
      {ToRads(120_r), ToRads(120_r)},
      {ToRads(180_r), ToRads(180_r)},
      {ToRads(240_r), ToRads(-120_r)},
      {ToRads(270_r), ToRads(-90_r)},
      {ToRads(345_r), ToRads(-15_r)},
      {ToRads(360_r), ToRads(0_r)},
      {ToRads(-15_r), ToRads(-15_r)},
      {ToRads(-90_r), ToRads(-90_r)},
      {ToRads(-120_r), ToRads(-120_r)},
      {ToRads(-180_r), ToRads(-180_r)},
      {ToRads(-240_r), ToRads(120_r)},
      {ToRads(-270_r), ToRads(90_r)},
      {ToRads(-345_r), ToRads(15_r)},
      {ToRads(-360_r), ToRads(0_r)},
  };

  using FQ = Quaternion(real);
  std::vector<FQ*> quatFs = {quatx, quaty, quatz};
  for (auto quatF : quatFs) {
    testCVal(quatF(0_r), quatF(0_r), quatF(0_r), {}); // Rest conditions
    for (auto angle : angles) {
      testCVal(quatF(angle[0]), kqI, kqI, quatF(angle[1]).ToRotationVector()); // rs -> r = wrap(rs)
      testCVal( // rt -> r = wrap(Invert(rs))
          kqI,
          quatF(angle[0]),
          kqI,
          quatF(-angle[1]).ToRotationVector());
      testCVal(kqI, kqI, quatF(angle[0]), quatF(angle[1]).ToRotationVector()); // rl -> r = wrap(rl)
      testCVal( // Equivalent rs and rt -> r = wrap(rs * Invert(rt)) -> I
          quatF(angle[0]),
          quatF(angle[1]),
          kqI,
          {});
      testCVal( // Cancelling rt and rl -> r = wrap(rl * Invert(rt)) -> I
          kqI,
          quatF(angle[0]),
          quatF(angle[1]),
          {});
    }
  }
}

// In corner cases (e.g. when input angle is +-pi, finite differences may flag a false negative
// because of the wrap around +-pi. We take the smallest finite difference (central, plus or minus)
static ColumnVector<real, 3> SelectClosest(
    ColumnVectorView<real const, 3> val0,
    ColumnVectorView<real const, 3> valp,
    ColumnVectorView<real const, 3> valm,
    RowMatrixView<real, 3, 1, krylov::kDynamic> jac,
    real delta) {
  auto cd = (valp - val0) * (1_r / (2_r * delta));
  auto pd = (valp - val0) * (1_r / delta);
  auto md = (val0 - valm) * (1_r / delta);
  ColumnVector<real, 3> cdDiff = cd - jac;
  ColumnVector<real, 3> pdDiff = pd - jac;
  ColumnVector<real, 3> mdDiff = md - jac;
  if (cdDiff.Norm() <= pdDiff.Norm() && cdDiff.Norm() <= mdDiff.Norm()) {
    return cd;
  } else if (pdDiff.Norm() <= cdDiff.Norm() && pdDiff.Norm() <= mdDiff.Norm()) {
    return pd;
  } else {
    return md;
  }
}

TEST(Constraints, RotationJacobian) {
  // Define method to evaluate constraint Jacobian
  // NOTE: Because we are using clamped rotations, general finite differences may
  // lead to large errors if the delta triggers the clamping. We get around this
  // issue by checking the derivative estimate with +delta and -delta and picking
  // the one that produces the smallest error.
  auto testFD = [](Quaternion qs, Quaternion const& qt, Quaternion const& ql) {
    real constexpr kDelta = 1.0e-2_r;
    real constexpr kTol = 1.0e-2_r;
    ColumnVector<real, 3> val;
    RowMatrix<real, 3, 3> Jac;

    EvalRotationFixedConstraint(qs, ql, qt, &val, &Jac, nullptr);

    auto val0 = val.Duplicate();
    auto JacFD = Jac.Duplicate();
    for (int i = 0; i < 3; ++i) {
      Real3 delta{};
      delta[i] = kDelta;
      Quaternion qsFD = Quaternion::FromRotationVector(delta) * qs;
      EvalRotationFixedConstraint(qsFD, ql, qt, &val, nullptr, nullptr);
      auto valp = val.Duplicate();

      delta[i] = -kDelta;
      qsFD = Quaternion::FromRotationVector(delta) * qs;
      EvalRotationFixedConstraint(qsFD, ql, qt, &val, nullptr, nullptr);
      auto valm = val.Duplicate();

      JacFD.Col(i) = SelectClosest(val0, valp, valm, Jac.Col(i), kDelta);
    }
    EXPECT_LT((RowMatrix<real, 3, 3>(JacFD - Jac)).Norm(), kTol * JacFD.Norm());
  };

  std::vector<real> angles = {
      ToRads(0_r),
      ToRads(15_r),
      ToRads(90_r),
      ToRads(120_r),
      ToRads(179_r),
      ToRads(240_r),
      ToRads(270_r),
      ToRads(345_r),
      ToRads(359_r),
      ToRads(-15_r),
      ToRads(-90_r),
      ToRads(-120_r),
      ToRads(-179_r),
      ToRads(-240_r),
      ToRads(-270_r),
      ToRads(-345_r),
      ToRads(-359_r)};
  std::vector<Real3> axes = {kvx, kvy, kvz, Normalize(kv123)};
  Quaternion ql = Quaternion::FromAxisAngle(Normalize(kv123), ToRads(15_r));
  for (auto angle_s : angles) {
    for (auto axis_s : axes) {
      Quaternion qs = Quaternion::FromAxisAngle(axis_s, angle_s);
      for (auto angle_t : angles) {
        for (auto axis_t : axes) {
          Quaternion qt = Quaternion::FromAxisAngle(axis_t, angle_t);
          testFD(qs, qt, ql);
        }
      }
    }
  }
}

TEST(Constraints, RotationJacobianTarget) {
  auto testFD = [](Quaternion qs, Quaternion const& qt, Quaternion const& ql) {
    real constexpr kDelta = 1.0e-2_r;
    real constexpr kTol = 1.0e-2_r;
    ColumnVector<real, 3> val;
    RowMatrix<real, 3, 3> Jac;

    EvalRotationFixedConstraint(qs, ql, qt, &val, nullptr, &Jac);

    auto val0 = val.Duplicate();
    auto JacFD = Jac.Duplicate();
    for (int i = 0; i < 3; ++i) {
      Real3 delta{};
      delta[i] = kDelta;
      Quaternion qtFD = Quaternion::FromRotationVector(delta) * qt;
      EvalRotationFixedConstraint(qs, ql, qtFD, &val, nullptr, nullptr);
      auto valp = val.Duplicate();

      delta[i] = -kDelta;
      qtFD = Quaternion::FromRotationVector(delta) * qt;
      EvalRotationFixedConstraint(qs, ql, qtFD, &val, nullptr, nullptr);
      auto valm = val.Duplicate();

      JacFD.Col(i) = SelectClosest(val0, valp, valm, Jac.Col(i), kDelta);
    }
    EXPECT_LT((RowMatrix<real, 3, 3>(JacFD - Jac)).Norm(), kTol * JacFD.Norm());
  };

  std::vector<real> angles = {
      ToRads(0_r),
      ToRads(15_r),
      ToRads(90_r),
      ToRads(120_r),
      ToRads(179_r),
      ToRads(240_r),
      ToRads(270_r),
      ToRads(345_r),
      ToRads(359_r),
      ToRads(-15_r),
      ToRads(-90_r),
      ToRads(-120_r),
      ToRads(-179_r),
      ToRads(-240_r),
      ToRads(-270_r),
      ToRads(-345_r),
      ToRads(-359_r)};
  std::vector<Real3> axes = {kvx, kvy, kvz, Normalize(kv123)};
  Quaternion ql = Quaternion::FromAxisAngle(Normalize(kv123), ToRads(15_r));
  for (auto angle_s : angles) {
    for (auto axis_s : axes) {
      Quaternion qs = Quaternion::FromAxisAngle(axis_s, angle_s);
      for (auto angle_t : angles) {
        for (auto axis_t : axes) {
          Quaternion qt = Quaternion::FromAxisAngle(axis_t, angle_t);
          testFD(qs, qt, ql);
        }
      }
    }
  }
}

/// EvalRigidSphericalJointConstraint ---------------------------------------------
TEST(Constraints, RigidSphericalJointValue) {
  Real3 localPosA = kvx;
  Real3 localPosB = -kvx;
  Real3 rA = localPosA;
  Real3 rB = localPosB;

  // Define method to evaluate constraint value
  auto testCVal = [&](Real3 tA, Quaternion qA, Real3 tB, Quaternion qB, Real3 cVal) {
    real constexpr kTol = 1.0e-3_r;
    ColumnVector<real, 3> val;

    TransformRT rbStateA(qA, tA);
    TransformRT rbStateB(qB, tB);
    EvalRigidSphericalJointConstraint(rbStateA, rbStateB, localPosA, localPosB, &val, nullptr);

    for (int i = 0; i < 3; ++i) {
      EXPECT_NEAR_RTOL(cVal[i], val[i], kTol);
    }
  };

  std::vector<Real3> vs = {kv0, kvx, kvy, kvz, kv123};
  std::vector<Quaternion> qs = {
      kqI,
      quatx(ToRads(90_r)),
      quatx(ToRads(180_r)),
      quatx(ToRads(-90_r)),
      quatx(ToRads(-360_r)),
      quaty(ToRads(90_r)),
      quaty(ToRads(180_r)),
      quaty(ToRads(-90_r)),
      quaty(ToRads(-360_r)),
      quatz(ToRads(90_r)),
      quatz(ToRads(180_r)),
      quatz(ToRads(-90_r)),
      quatz(ToRads(-360_r)),
  };

  for (auto tA : vs) {
    for (auto qA : qs) {
      for (auto tB : vs) {
        for (auto qB : qs) {
          testCVal(
              tA,
              qA,
              tB,
              qB,
              TransformRT(qA, tA).TransformPoint(rA) - TransformRT(qB, tB).TransformPoint(rB));
        }
      }
    }
  }
}

TEST(Constraints, RigidSphericalJointJacobian) {
  Real3 localPosA = kvx;
  Real3 localPosB = -kvx;

  // Define method to evaluate constraint Jacobian
  auto testFD = [&](Real3 tA, Quaternion qA, Real3 tB, Quaternion qB) {
    real constexpr kDelta = 1e-2_r;
    real constexpr kTol = 3e-2_r;
    ColumnVector<real, 3> val;
    RowMatrix<real, 3, 12> Jac;

    TransformRT rbStateA(qA, tA);
    TransformRT rbStateB(qB, tB);
    EvalRigidSphericalJointConstraint(rbStateA, rbStateB, localPosA, localPosB, &val, &Jac);

    auto JacFD = Jac.Duplicate();

    for (int i = 0; i < 3; ++i) { // dC/dxA
      Real3 xA = tA;
      xA[i] += kDelta;
      rbStateA.SetTranslation(xA);
      EvalRigidSphericalJointConstraint(rbStateA, rbStateB, localPosA, localPosB, &val, nullptr);
      auto valp = val.Duplicate();

      xA[i] -= 2_r * kDelta;
      rbStateA.SetTranslation(xA);
      EvalRigidSphericalJointConstraint(rbStateA, rbStateB, localPosA, localPosB, &val, nullptr);
      auto valm = val.Duplicate();

      JacFD.Col(i) = (valp - valm) * (1_r / (2_r * kDelta));
    }
    rbStateA.SetTranslation(tA);
    for (int i = 0; i < 3; ++i) { // dC/dRA
      Real3 delta{};
      delta[i] = kDelta;
      rbStateA.SetRotation(Quaternion::FromRotationVector(delta) * qA);
      EvalRigidSphericalJointConstraint(rbStateA, rbStateB, localPosA, localPosB, &val, nullptr);
      auto valp = val.Duplicate();

      delta[i] = -kDelta;
      rbStateA.SetRotation(Quaternion::FromRotationVector(delta) * qA);
      EvalRigidSphericalJointConstraint(rbStateA, rbStateB, localPosA, localPosB, &val, nullptr);
      auto valm = val.Duplicate();

      JacFD.Col(i + 3) = (valp - valm) * (1_r / (2_r * kDelta));
    }
    rbStateA.SetRotation(qA);
    for (int i = 0; i < 3; ++i) { // dC/dxB
      Real3 xB = tB;
      xB[i] += kDelta;
      rbStateB.SetTranslation(xB);
      EvalRigidSphericalJointConstraint(rbStateA, rbStateB, localPosA, localPosB, &val, nullptr);
      auto valp = val.Duplicate();

      xB[i] -= 2_r * kDelta;
      rbStateB.SetTranslation(xB);
      EvalRigidSphericalJointConstraint(rbStateA, rbStateB, localPosA, localPosB, &val, nullptr);
      auto valm = val.Duplicate();

      JacFD.Col(i + 6) = (valp - valm) * (1_r / (2_r * kDelta));
    }
    rbStateB.SetTranslation(tB);
    for (int i = 0; i < 3; ++i) { // dC/dRB
      Real3 delta{};
      delta[i] = kDelta;
      rbStateB.SetRotation(Quaternion::FromRotationVector(delta) * qB);
      EvalRigidSphericalJointConstraint(rbStateA, rbStateB, localPosA, localPosB, &val, nullptr);
      auto valp = val.Duplicate();

      delta[i] = -kDelta;
      rbStateB.SetRotation(Quaternion::FromRotationVector(delta) * qB);
      EvalRigidSphericalJointConstraint(rbStateA, rbStateB, localPosA, localPosB, &val, nullptr);
      auto valm = val.Duplicate();

      JacFD.Col(i + 9) = (valp - valm) * (1_r / (2_r * kDelta));
    }
    rbStateB.SetRotation(qB);
    EXPECT_LE((RowMatrix<real, 3, 12>(JacFD - Jac)).Norm(), kTol * JacFD.Norm());
  };

  std::vector<Real3> vs = {kv0, kvy + kvz, kv123};
  std::vector<real> angles = {ToRads(0_r), ToRads(-15_r), ToRads(179_r), ToRads(359_r)};
  std::vector<Real3> axes = {kvx, Normalize(kvy + kvz), Normalize(kv123)};
  for (auto tA : vs) {
    for (auto angleA : angles) {
      for (auto axisA : axes) {
        Quaternion qA = Quaternion::FromAxisAngle(axisA, angleA);
        for (auto tB : vs) {
          for (auto angleB : angles) {
            for (auto axisB : axes) {
              Quaternion qB = Quaternion::FromAxisAngle(axisB, angleB);
              testFD(tA, qA, tB, qB);
            }
          }
        }
      }
    }
  }
}

/// RigidPrismaticJointConstraint ---------------------------------------------
static Real3 EvalRelativeTranslation(
    Real3 const& tA,
    Quaternion const& qA,
    Real3 const& tB,
    VMatrix3x3r const& localFrame) {
  auto trel = Conjugate(qA) * (tB - tA);
  return ToReal3(DotMatVec3x3(localFrame, ToSimd(trel)));
}

static std::tuple<Quaternion, Real3, real, real> SetupRigidPrismaticJointTest() {
  // Initialize data
  Real3 localAxis{3_r, -2_r, 4_r};
  localAxis = Normalize(localAxis);
  Real3 axis = Cross(localAxis, kvz);
  real normSqr = NormSqr(axis);
  if (normSqr > 1e-6) {
    real angle = std::acos(Dot(localAxis, kvz));
    axis = axis / std::sqrt(normSqr) * angle;
  }
  Quaternion localFrame = Quaternion::FromRotationVector(axis);
  VMatrix3x3r localFrameMatrix = ToVMatrix3x3(localFrame);

  TransformRT restA(
      Quaternion::FromRotationVector(Real3{0.1_r, -0.2_r, 0.1_r}), Real3{2_r, 1_r, 2_r});
  TransformRT restB(
      Quaternion::FromRotationVector(Real3{-0.2_r, 0.1_r, -0.1_r}), Real3{-2_r, 0_r, 1_r});

  Real3 tref = EvalRelativeTranslation(
      restA.GetTranslation(), restA.GetRotation(), restB.GetTranslation(), localFrameMatrix);

  static real constexpr kMaxLimit = 2.2_r;
  static real constexpr kMinLimit = -1_r;

  return {localFrame, tref, kMaxLimit, kMinLimit};
}

TEST(Constraints, RigidPrismaticJointValue) {
  auto setup = SetupRigidPrismaticJointTest();
  Quaternion localFrame = std::get<0>(setup);
  Real3 tRef = std::get<1>(setup);
  real maxLimit = std::get<2>(setup);
  real minLimit = std::get<3>(setup);

  // Define method to test constraint value
  auto testCVal = [&](Real3 const& tA, Quaternion const& qA, Real3 const& tB, Real3 cVal) {
    if (cVal[2] > maxLimit) {
      cVal[2] -= maxLimit;
    } else if (cVal[2] < minLimit) {
      cVal[2] -= minLimit;
    } else {
      cVal[2] = 0_r; // Translation on the free axis is ignored
    }

    real constexpr kTol = 1.0e-3_r;
    ColumnVector<real, 3> val;

    TransformRT rbStateA(qA, tA);

    EvalRigidPrismaticJointConstraint(
        rbStateA, tB, localFrame, tRef, maxLimit, minLimit, &val, nullptr);

    for (int i = 0; i < 3; ++i) {
      EXPECT_NEAR_RTOL(cVal[i], val[i], kTol);
    }
  };

  std::vector<Real3> vs = {kv0, kvx, kvy, kvz, -kv123};
  std::vector<Quaternion> qs = {
      kqI,
      quatx(ToRads(90_r)),
      quatx(ToRads(180_r)),
      quatx(ToRads(-90_r)),
      quatx(ToRads(-360_r)),
      quaty(ToRads(90_r)),
      quaty(ToRads(180_r)),
      quaty(ToRads(-90_r)),
      quaty(ToRads(-360_r)),
      quatz(ToRads(90_r)),
      quatz(ToRads(180_r)),
      quatz(ToRads(-90_r)),
      quatz(ToRads(-360_r)),
  };

  VMatrix3x3r localFrameMatrix = ToVMatrix3x3(localFrame);
  for (auto tA : vs) {
    for (auto qA : qs) {
      for (auto tB : vs) {
        testCVal(tA, qA, tB, EvalRelativeTranslation(tA, qA, tB, localFrameMatrix) - tRef);
      }
    }
  }
}

TEST(Constraints, RigidPrismaticJointJacobian) {
  auto setup = SetupRigidPrismaticJointTest();
  Quaternion localFrame = std::get<0>(setup);
  Real3 tRef = std::get<1>(setup);
  real maxLimit = std::get<2>(setup);
  real minLimit = std::get<3>(setup);

  // Define method to evaluate constraint Jacobian
  auto testFD = [&](Real3 tA, Quaternion qA, Real3 tB) {
    real constexpr kDelta = 1.0e-2_r;
    real constexpr kTol = 1.0e-2_r;
    ColumnVector<real, 3> val;
    RowMatrix<real, 3, 9> Jac;

    TransformRT rbStateA(qA, tA);
    EvalRigidPrismaticJointConstraint(
        rbStateA, tB, localFrame, tRef, maxLimit, minLimit, &val, &Jac);

    auto Jac_FD = Jac.Duplicate();

    for (int i = 0; i < 3; ++i) { // dC/dxA
      Real3 xA = tA;
      xA[i] += kDelta;
      rbStateA.SetTranslation(xA);
      EvalRigidPrismaticJointConstraint(
          rbStateA, tB, localFrame, tRef, maxLimit, minLimit, &val, nullptr);
      auto valp = val.Duplicate();

      xA[i] -= 2_r * kDelta;
      rbStateA.SetTranslation(xA);
      EvalRigidPrismaticJointConstraint(
          rbStateA, tB, localFrame, tRef, maxLimit, minLimit, &val, nullptr);
      auto valm = val.Duplicate();

      Jac_FD.Col(i) = (valp - valm) * (1_r / (2_r * kDelta));
    }
    rbStateA.SetTranslation(tA);
    for (int i = 0; i < 3; ++i) { // dC/dRA
      Real3 delta{};
      delta[i] = kDelta;
      rbStateA.SetRotation(Quaternion::FromRotationVector(delta) * qA);
      EvalRigidPrismaticJointConstraint(
          rbStateA, tB, localFrame, tRef, maxLimit, minLimit, &val, nullptr);
      auto valp = val.Duplicate();

      delta[i] = -kDelta;
      rbStateA.SetRotation(Quaternion::FromRotationVector(delta) * qA);
      EvalRigidPrismaticJointConstraint(
          rbStateA, tB, localFrame, tRef, maxLimit, minLimit, &val, nullptr);
      auto valm = val.Duplicate();

      Jac_FD.Col(i + 3) = (valp - valm) * (1_r / (2_r * kDelta));
    }
    rbStateA.SetRotation(qA);
    for (int i = 0; i < 3; ++i) { // dC/dxB
      Real3 xB = tB;
      xB[i] += kDelta;
      EvalRigidPrismaticJointConstraint(
          rbStateA, xB, localFrame, tRef, maxLimit, minLimit, &val, nullptr);
      auto valp = val.Duplicate();

      xB[i] -= 2_r * kDelta;
      EvalRigidPrismaticJointConstraint(
          rbStateA, xB, localFrame, tRef, maxLimit, minLimit, &val, nullptr);
      auto valm = val.Duplicate();

      Jac_FD.Col(i + 6) = (valp - valm) * (1_r / (2_r * kDelta));
    }
    EXPECT_LE((RowMatrix<real, 3, 9>(Jac_FD - Jac)).Norm(), kTol * Jac_FD.Norm());
  };

  std::vector<Real3> vs = {kv0, kvy + kvz, -kv123};
  std::vector<real> angles = {ToRads(0_r), ToRads(-15_r), ToRads(179_r), ToRads(359_r)};
  std::vector<Real3> axes = {kvx, Normalize(kvy + kvz), Normalize(kv123)};
  for (auto tA : vs) {
    for (auto angleA : angles) {
      for (auto axisA : axes) {
        Quaternion qA = Quaternion::FromAxisAngle(axisA, angleA);
        for (auto tB : vs) {
          testFD(tA, qA, tB);
        }
      }
    }
  }
}

/// EvalDeformableNodeToDeformableNodeConstraint ---------------------------------------------
TEST(Constraints, DeformableNodeToDeformableNodeValue) {
  // Define method to evaluate constraint
  auto testCVal = [&](Real3 const& dispA,
                      Real3 const& tA,
                      Real3 const& rA,
                      Quaternion const& qA,
                      Real3 const& dispB,
                      Real3 const& tB,
                      Real3 const& rB,
                      Quaternion const& qB,
                      Real3 const& cVal) {
    TransformRT txA(qA, tA);
    TransformRT txB(qB, tB);

    ColumnVector<real, 3> val;
    EvalDeformableNodeToDeformableNodeConstraint(txA, txB, rA + dispA, rB + dispB, &val, nullptr);

    real constexpr kTol = 1.0e-3_r;
    for (int i = 0; i < 3; ++i) {
      EXPECT_NEAR_RTOL(cVal[i], val[i], kTol);
    }
  };

  // Manual tests
  testCVal(kv0, kv0, kv123, kqI, kv0, kv0, -kv123, kqI, 2_r * kv123);
  testCVal(-kv123, kv0, kv123, kqI, kv0, kv0, -kv123, kqI, kv123);
  testCVal(-kv123, kv0, kv123, kqI, kv123, kv0, -kv123, kqI, kv0);
  testCVal(-kv123, kv123, kv123, kqI, kv123, -kv123, -kv123, kqI, 2_r * kv123);
  testCVal(kvx, kv0, kv123, kqI, kv0, kv0, -kv123, kqI, kvx + 2_r * kv123);
  testCVal(kvy, kv0, kv123, kqI, kv0, kv0, -kv123, kqI, kvy + 2_r * kv123);
  testCVal(kvz, kv0, kv123, kqI, kv0, kv0, -kv123, kqI, kvz + 2_r * kv123);
  testCVal(kv0, kv0, kv123, kqI, kvx, kv0, -kv123, kqI, -kvx + 2_r * kv123);
  testCVal(kv0, kv0, kv123, kqI, kvy, kv0, -kv123, kqI, -kvy + 2_r * kv123);
  testCVal(kv0, kv0, kv123, kqI, kvz, kv0, -kv123, kqI, -kvz + 2_r * kv123);
  testCVal(kv0, kvx, kv123, kqI, kv0, kv0, -kv123, kqI, kvx + 2_r * kv123);
  testCVal(kv0, kvy, kv123, kqI, kv0, kv0, -kv123, kqI, kvy + 2_r * kv123);
  testCVal(kv0, kvz, kv123, kqI, kv0, kv0, -kv123, kqI, kvz + 2_r * kv123);
  testCVal(kv0, kv0, kv123, kqI, kv0, kvx, -kv123, kqI, -kvx + 2_r * kv123);
  testCVal(kv0, kv0, kv123, kqI, kv0, kvy, -kv123, kqI, -kvy + 2_r * kv123);
  testCVal(kv0, kv0, kv123, kqI, kv0, kvz, -kv123, kqI, -kvz + 2_r * kv123);
  testCVal(kv0, kv0, kv123, quatx(ToRads(180_r)), kv0, kv0, -kv123, kqI, 2_r * kvx);
  testCVal(kv0, kv0, kv123, quaty(ToRads(180_r)), kv0, kv0, -kv123, kqI, 2_r * 2_r * kvy);
  testCVal(kv0, kv0, kv123, quatz(ToRads(180_r)), kv0, kv0, -kv123, kqI, 2_r * 3_r * kvz);
  testCVal(kv0, kv0, kv123, kqI, kv0, kv0, -kv123, quatx(ToRads(180_r)), 2_r * kvx);
  testCVal(kv0, kv0, kv123, kqI, kv0, kv0, -kv123, quaty(ToRads(180_r)), 2_r * 2_r * kvy);
  testCVal(kv0, kv0, kv123, kqI, kv0, kv0, -kv123, quatz(ToRads(180_r)), 2_r * 3_r * kvz);
  testCVal(kv0, kvx, kv123, quatx(ToRads(180_r)), kv0, kv0, -kv123, kqI, 2_r * kvx + kvx);
  testCVal(kv0, kvx, kv123, quaty(ToRads(180_r)), kv0, kv0, -kv123, kqI, 2_r * 2_r * kvy + kvx);
  testCVal(kv0, kvx, kv123, quatz(ToRads(180_r)), kv0, kv0, -kv123, kqI, 2_r * 3_r * kvz + kvx);
  testCVal(kv0, kv0, kv123, kqI, kv0, kvx, -kv123, quatx(ToRads(180_r)), 2_r * kvx - kvx);
  testCVal(kv0, kv0, kv123, kqI, kv0, kvx, -kv123, quaty(ToRads(180_r)), 2_r * 2_r * kvy - kvx);
  testCVal(kv0, kv0, kv123, kqI, kv0, kvx, -kv123, quatz(ToRads(180_r)), 2_r * 3_r * kvz - kvx);
  testCVal(kvx, kv0, kv123, quatx(ToRads(180_r)), kv0, kv0, -kv123, kqI, 2_r * kvx + kvx);
  testCVal(kvy, kv0, kv123, quatx(ToRads(180_r)), kv0, kv0, -kv123, kqI, 2_r * kvx - kvy);
  testCVal(kvz, kv0, kv123, quatx(ToRads(180_r)), kv0, kv0, -kv123, kqI, 2_r * kvx - kvz);
  testCVal(kvx, kv0, kv123, quaty(ToRads(180_r)), kv0, kv0, -kv123, kqI, 2_r * 2_r * kvy - kvx);
  testCVal(kvy, kv0, kv123, quaty(ToRads(180_r)), kv0, kv0, -kv123, kqI, 2_r * 2_r * kvy + kvy);
  testCVal(kvz, kv0, kv123, quaty(ToRads(180_r)), kv0, kv0, -kv123, kqI, 2_r * 2_r * kvy - kvz);
  testCVal(kvx, kv0, kv123, quatz(ToRads(180_r)), kv0, kv0, -kv123, kqI, 2_r * 3_r * kvz - kvx);
  testCVal(kvy, kv0, kv123, quatz(ToRads(180_r)), kv0, kv0, -kv123, kqI, 2_r * 3_r * kvz - kvy);
  testCVal(kvz, kv0, kv123, quatz(ToRads(180_r)), kv0, kv0, -kv123, kqI, 2_r * 3_r * kvz + kvz);
  testCVal(kv0, kv0, kv123, kqI, kvx, kv0, -kv123, quatx(ToRads(180_r)), 2_r * kvx - kvx);
  testCVal(kv0, kv0, kv123, kqI, kvy, kv0, -kv123, quatx(ToRads(180_r)), 2_r * kvx + kvy);
  testCVal(kv0, kv0, kv123, kqI, kvz, kv0, -kv123, quatx(ToRads(180_r)), 2_r * kvx + kvz);
  testCVal(kv0, kv0, kv123, kqI, kvx, kv0, -kv123, quaty(ToRads(180_r)), 2_r * 2_r * kvy + kvx);
  testCVal(kv0, kv0, kv123, kqI, kvy, kv0, -kv123, quaty(ToRads(180_r)), 2_r * 2_r * kvy - kvy);
  testCVal(kv0, kv0, kv123, kqI, kvz, kv0, -kv123, quaty(ToRads(180_r)), 2_r * 2_r * kvy + kvz);
  testCVal(kv0, kv0, kv123, kqI, kvx, kv0, -kv123, quatz(ToRads(180_r)), 2_r * 3_r * kvz + kvx);
  testCVal(kv0, kv0, kv123, kqI, kvy, kv0, -kv123, quatz(ToRads(180_r)), 2_r * 3_r * kvz + kvy);
  testCVal(kv0, kv0, kv123, kqI, kvz, kv0, -kv123, quatz(ToRads(180_r)), 2_r * 3_r * kvz - kvz);

  // Systematic tests
  std::vector<Real3> vs = {kv0, kvx, kvy, kvz, kvx + kvy, kvx + kvz, kvx + kvy + kvz};
  std::vector<Real3> ts = {kv0, kv123};
  std::vector<Real3> rs = {kv123, -kv123};
  std::vector<Quaternion> qs = {
      kqI,
      quatx(ToRads(60_r)),
      quatx(ToRads(-120_r)),
      quaty(ToRads(180_r)),
      quaty(ToRads(-240_r)),
      quatz(ToRads(300_r)),
  };
  for (auto uA : vs) {
    for (auto tA : ts) {
      for (auto rA : rs) {
        for (auto qA : qs) {
          for (auto uB : vs) {
            for (auto tB : ts) {
              for (auto rB : rs) {
                for (auto qB : qs) {
                  Real3 cVal = TransformRT(qA, tA).TransformPoint(rA + uA) -
                      TransformRT(qB, tB).TransformPoint(rB + uB);
                  testCVal(uA, tA, rA, qA, uB, tB, rB, qB, cVal);
                }
              }
            }
          }
        }
      }
    }
  }
}

TEST(Constraints, DeformableNodeToDeformableNodeJacobian) {
  // Define method to evaluate Jacobian through finite differences
  auto testFD = [](Real3 dispA,
                   Real3 const& tA,
                   Real3 const& rA,
                   Quaternion const& qA,
                   Real3 dispB,
                   Real3 const& tB,
                   Real3 const& rB,
                   Quaternion const& qB) {
    TransformRT txA(qA, tA);
    TransformRT txB(qB, tB);

    ColumnVector<real, 3> val;
    RowMatrix<real, 3, 6> Jac;
    EvalDeformableNodeToDeformableNodeConstraint(txA, txB, rA + dispA, rB + dispB, &val, &Jac);
    auto JacFD = Jac.Duplicate();

    real constexpr kDelta = 1.0e-2_r;
    real constexpr kTol = 1.0e-2_r;
    for (int i = 0; i < 3; ++i) { // dC/duA
      real v0 = dispA[i];
      real dx = std::max(kDelta, std::abs(kDelta * dispA[i]));
      dispA[i] += dx;
      EvalDeformableNodeToDeformableNodeConstraint(txA, txB, rA + dispA, rB + dispB, &val, nullptr);
      auto valp = val.Duplicate();

      dispA[i] -= 2_r * dx;
      EvalDeformableNodeToDeformableNodeConstraint(txA, txB, rA + dispA, rB + dispB, &val, nullptr);
      auto valm = val.Duplicate();
      dispA[i] = v0;

      JacFD.Col(i) = (valp - valm) * (1_r / (2_r * dx));
    }
    for (int i = 0; i < 3; ++i) { // dC/duB
      real v0 = dispB[i];
      real dx = std::max(kDelta, kDelta * dispB[i]);
      dispB[i] += dx;
      EvalDeformableNodeToDeformableNodeConstraint(txA, txB, rA + dispA, rB + dispB, &val, nullptr);
      auto valp = val.Duplicate();

      dispB[i] -= 2_r * dx;
      EvalDeformableNodeToDeformableNodeConstraint(txA, txB, rA + dispA, rB + dispB, &val, nullptr);
      auto valm = val.Duplicate();
      dispB[i] = v0;

      JacFD.Col(i + 3) = (valp - valm) * (1_r / (2_r * dx));
    }
    EXPECT_LE((RowMatrix<real, 3, 6>(JacFD - Jac)).Norm(), kTol * JacFD.Norm());
  };

  std::vector<Real3> vs = {kv0, kvx, kvy, kvz, kvx + kvy + kvz};
  std::vector<Real3> ts = {kv0, kv123};
  std::vector<Real3> rs = {kv123, -kv123};
  std::vector<Quaternion> qs = {
      kqI,
      quatx(ToRads(60_r)),
      quatx(ToRads(-120_r)),
      quaty(ToRads(180_r)),
      quaty(ToRads(-240_r)),
      quatz(ToRads(300_r)),
      quatz(ToRads(-360_r)),
  };
  for (auto uA : vs) {
    for (auto tA : ts) {
      for (auto rA : rs) {
        for (auto qA : qs) {
          for (auto uB : vs) {
            for (auto tB : ts) {
              for (auto rB : rs) {
                for (auto qB : qs) {
                  testFD(uA, tA, rA, qA, uB, tB, rB, qB);
                }
              }
            }
          }
        }
      }
    }
  }
}

/// EvalJointRotationRangeConstraint ---------------------------------------------
TEST(Constraints, JointRotationRangeValue) {
  // Define method to evaluate constraint value
  auto testCVal = [](Quaternion const& qA,
                     Quaternion const& qB,
                     Quaternion const& q0,
                     Quaternion const& qr,
                     Real2 rangeX,
                     Real2 rangeY,
                     Real2 rangeZ,
                     Real3 const& cVal) {
    real constexpr kTol = 1.0e-2_r;
    ColumnVector<real, 3> val;

    Real3 min = {rangeX[0], rangeY[0], rangeZ[0]};
    Real3 max = {rangeX[1], rangeY[1], rangeZ[1]};
    bool isActive{};
    EvalJointRotationRangeConstraint(qA, qB, q0, qr, min, max, &val, nullptr, isActive);

    for (int i = 0; i < 3; ++i) {
      EXPECT_NEAR_RTOL(cVal[i], val[i], kTol);
    }
  };

  // Arbritary angles that cover different ranges and combinations
  std::vector<Quaternion> q0s = {
      Quaternion::FromAxisAngle(kvx, ToRads(50_r)),
      Quaternion::FromAxisAngle(kvy, ToRads(125_r)),
      Quaternion::FromAxisAngle(Normalize(kv123), ToRads(-20_r))};
  std::vector<Quaternion> qrs = {
      Quaternion::FromAxisAngle(kvx, ToRads(-10_r)),
      Quaternion::FromAxisAngle(kvz, ToRads(66_r)),
      Quaternion::FromAxisAngle(Normalize(kv123), ToRads(-110_r))};
  std::vector<Real3> axes = {kvx, kvy, kvz, Normalize(kv123)};
  std::vector<real> angles = {ToRads(15_r), ToRads(105_r), ToRads(-105_r), ToRads(-15_r)};
  std::vector<real> diffs = {ToRads(30_r), ToRads(135_r), ToRads(-30_r), ToRads(-135_r)};
  Real2 range_X = {ToRads(-35_r), ToRads(20_r)}; // Non-trivial angle ranges
  Real2 range_Y = {ToRads(-125_r), ToRads(80_r)};
  Real2 range_Z = {ToRads(10_r), ToRads(160_r)};
  Real3 rmin = {range_X[0], range_Y[0], range_Z[0]};
  Real3 rmax = {range_X[1], range_Y[1], range_Z[1]};
  for (auto axis_A : axes) {
    for (auto angle_A : angles) {
      Quaternion qA = Quaternion::FromAxisAngle(axis_A, angle_A);
      for (auto axis_diff : axes) {
        for (auto angle_diff : diffs) {
          for (auto q0 : q0s) {
            for (auto qr : qrs) {
              Quaternion qdiff = Quaternion::FromAxisAngle(axis_diff, angle_diff);
              Quaternion qB = Normalize(qA * qdiff);
              Quaternion qAB = Normalize(q0.GetConjugate() * qdiff * q0 * qr.GetConjugate());
              Real3 rab = qAB.ToRotationVector();
              Real3 rab_rmax = rab - rmax;
              Real3 rmin_rab = rmin - rab;
              Real3 cVal = {0_r, 0_r, 0_r};
              for (int i = 0; i < 3; ++i) {
                if (rab_rmax[i] > 0_r) {
                  cVal[i] = rab_rmax[i];
                } else if (rmin_rab[i] > 0_r) {
                  cVal[i] = rmin_rab[i];
                }
              }
              testCVal(qA, qB, q0, qr, range_X, range_Y, range_Z, cVal);
            }
          }
        }
      }
    }
  }
}

TEST(Constraints, JointRotationRangeJacobian) {
  // Define method to evaluate constraint Jacobian
  auto testFD = [](Quaternion const& qA,
                   Quaternion const& qB,
                   Quaternion const& q0,
                   Quaternion const& qr,
                   Real2 rangeX,
                   Real2 rangeY,
                   Real2 rangeZ) {
    real constexpr kDelta = 1.0e-2_r;
    real constexpr kTol = 1.0e-2_r;

    ColumnVector<real, 3> val0;
    RowMatrix<real, 3, 6> Jac;

    Real3 min = {rangeX[0], rangeY[0], rangeZ[0]};
    Real3 max = {rangeX[1], rangeY[1], rangeZ[1]};
    bool isActive{};
    EvalJointRotationRangeConstraint(qA, qB, q0, qr, min, max, &val0, &Jac, isActive);

    auto JacFD = Jac.Duplicate();
    for (int i = 0; i < 3; ++i) {
      Real3 delta{};
      delta[i] = kDelta;
      auto qAp = Quaternion::FromRotationVector(delta) * qA;
      ColumnVector<real, 3> valp;
      EvalJointRotationRangeConstraint(qAp, qB, q0, qr, min, max, &valp, nullptr, isActive);

      delta[i] = -kDelta;
      auto qAm = Quaternion::FromRotationVector(delta) * qA;
      ColumnVector<real, 3> valm;
      EvalJointRotationRangeConstraint(qAm, qB, q0, qr, min, max, &valm, nullptr, isActive);

      JacFD.Col(i) = SelectClosest(val0, valp, valm, Jac.Col(i), kDelta);
    }

    for (int i = 0; i < 3; ++i) {
      Real3 delta{};
      delta[i] = kDelta;
      auto qBp = Quaternion::FromRotationVector(delta) * qB;
      ColumnVector<real, 3> valp;
      EvalJointRotationRangeConstraint(qA, qBp, q0, qr, min, max, &valp, nullptr, isActive);

      delta[i] = -kDelta;
      auto qBm = Quaternion::FromRotationVector(delta) * qB;
      ColumnVector<real, 3> valm;
      EvalJointRotationRangeConstraint(qA, qBm, q0, qr, min, max, &valm, nullptr, isActive);

      JacFD.Col(i + 3) = SelectClosest(val0, valp, valm, Jac.Col(i + 3), kDelta);
    }

    EXPECT_LE((RowMatrix<real, 3, 6>(JacFD - Jac)).Norm(), kTol * JacFD.Norm());
  };

  std::vector<Quaternion> q0s = {
      Quaternion::FromAxisAngle(kvx, ToRads(50_r)),
      Quaternion::FromAxisAngle(Normalize(kv123), ToRads(-20_r))};
  std::vector<Quaternion> qrs = {
      Quaternion::FromAxisAngle(kvx, ToRads(-10_r)),
      Quaternion::FromAxisAngle(Normalize(kv123), ToRads(-110_r))};
  std::vector<Real3> axes = {kvx, kvy, kvz, Normalize(kv123)}; // Angles for A in all 4 quadrants
  std::vector<real> angles = {
      ToRads(15_r), ToRads(105_r), ToRads(-105_r)}; // Angles for A in all 4 quadrants
  std::vector<real> diffs = {
      ToRads(30_r), ToRads(135_r), ToRads(-30_r), ToRads(-135_r)}; // Angle differences in [-pi,pi]
  Real2 range_X = {ToRads(-35_r), ToRads(20_r)};
  Real2 range_Y = {ToRads(-125_r), ToRads(80_r)};
  Real2 range_Z = {ToRads(10_r), ToRads(160_r)};
  Real3 rmin = {range_X[0], range_Y[0], range_Z[0]};
  Real3 rmax = {range_X[1], range_Y[1], range_Z[1]};
  for (auto axis_A : axes) {
    for (auto angle_A : angles) {
      Quaternion qA = Quaternion::FromAxisAngle(axis_A, angle_A);
      for (auto axis_diff : axes) {
        for (auto angle_diff : diffs) {
          for (auto q0 : q0s) {
            for (auto qr : qrs) {
              Quaternion qdiff = Quaternion::FromAxisAngle(axis_diff, angle_diff);
              Quaternion qB = Normalize(qA * qdiff);
              testFD(qA, qB, q0, qr, range_X, range_Y, range_Z);
            }
          }
        }
      }
    }
  }
}

/// EvalJointRotationTargetConstraint ---------------------------------------------
TEST(Constraints, JointRotationTargetValue) {
  // Define method to evaluate constraint value
  auto testCVal = [](Quaternion const& qA,
                     Quaternion const& qB,
                     Quaternion const& qAr,
                     Quaternion const& qBr,
                     Quaternion const& q0,
                     Real3 const& cVal) {
    auto qr = qAr.GetConjugate() * qBr;

    ColumnVector<real, 3> val;
    EvalJointRotationTargetConstraint(qA, qB, q0, qr, &val, nullptr, nullptr);

    real constexpr kTol = 1.0e-2_r;
    for (int i = 0; i < 3; ++i) {
      EXPECT_NEAR_RTOL(cVal[i], val[i], kTol);
    }
  };

  // Arbritary angles that cover different ranges and combinations
  std::vector<Quaternion> q0s = {
      Quaternion::FromAxisAngle(kvx, ToRads(50_r)),
      Quaternion::FromAxisAngle(Normalize(kv123), ToRads(-20_r))};
  std::vector<Real3> axes = {kvx, kvy, kvz, Normalize(kv123)};
  std::vector<real> angles = {ToRads(15_r), ToRads(-105_r)};
  std::vector<real> diffs = {ToRads(30_r), ToRads(-135_r)};
  for (auto q0 : q0s) {
    for (auto axis_A : axes) {
      for (auto angle_A : angles) {
        Quaternion qA = Quaternion::FromAxisAngle(axis_A, angle_A);
        for (auto axis_Ar : axes) {
          for (auto angle_Ar : angles) {
            Quaternion qAr = Quaternion::FromAxisAngle(axis_Ar, angle_Ar);
            for (auto axis_d : axes) {
              for (auto angle_d : diffs) {
                Quaternion qd = Quaternion::FromAxisAngle(axis_d, angle_d);
                for (auto axis_diffr : axes) {
                  for (auto angle_diffr : diffs) {
                    Quaternion qdiffr = Quaternion::FromAxisAngle(axis_diffr, angle_diffr);
                    Quaternion qdiff = qdiffr * qd; // qdiff = qdiffr * qd
                    Quaternion qB = Normalize(qA * qdiff); // qB = qA * qdiff = qA * qdiffr * qd
                    Quaternion qBr = Normalize(qAr * qdiffr); // qBr = qAr * qdiffr
                    Real3 rd = Normalize(q0.GetConjugate() * qd * q0).ToRotationVector();
                    testCVal(qA, qB, qAr, qBr, q0, rd);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

TEST(Constraints, JointRotationTargetJacobian) {
  // Define method to evaluate constraint Jacobian
  auto testFD = [](Quaternion const& qA,
                   Quaternion const& qB,
                   Quaternion const& qAr,
                   Quaternion const& qBr,
                   Quaternion const& q0) {
    real constexpr kDelta = 1.0e-2_r;
    real constexpr kTol = 1.0e-2_r;

    auto qr = qAr.GetConjugate() * qBr;

    ColumnVector<real, 3> val;
    RowMatrix<real, 3, 6> Jac;
    EvalJointRotationTargetConstraint(qA, qB, q0, qr, &val, &Jac, nullptr);

    auto JacFD = Jac.Duplicate();
    for (int i = 0; i < 3; ++i) {
      Real3 delta{};
      delta[i] = kDelta;
      auto qAp = Quaternion::FromRotationVector(delta) * qA;
      ColumnVector<real, 3> valp;
      EvalJointRotationTargetConstraint(qAp, qB, q0, qr, &valp, nullptr, nullptr);
      delta[i] = -kDelta;
      auto qAm = Quaternion::FromRotationVector(delta) * qA;
      ColumnVector<real, 3> valm;
      EvalJointRotationTargetConstraint(qAm, qB, q0, qr, &valm, nullptr, nullptr);

      JacFD.Col(i) = SelectClosest(val, valp, valm, Jac.Col(i), kDelta);
    }

    for (int i = 0; i < 3; ++i) {
      Real3 delta{};
      delta[i] = kDelta;
      auto qBp = Quaternion::FromRotationVector(delta) * qB;
      ColumnVector<real, 3> valp;
      EvalJointRotationTargetConstraint(qA, qBp, q0, qr, &valp, nullptr, nullptr);

      delta[i] = -kDelta;
      auto qBm = Quaternion::FromRotationVector(delta) * qB;
      ColumnVector<real, 3> valm;
      EvalJointRotationTargetConstraint(qA, qBm, q0, qr, &valm, nullptr, nullptr);

      JacFD.Col(i + 3) = SelectClosest(val, valp, valm, Jac.Col(i + 3), kDelta);
    }

    EXPECT_LE((RowMatrix<real, 3, 6>(JacFD - Jac)).Norm(), kTol * JacFD.Norm());
  };

  // Arbritary angles that cover different ranges and combinations
  // Note that finite differences will fail if any rd_i is zero because every component of
  // c = abs(rd) is non-differentiable at rd_i = 0. We ensure all test combinations produce
  // non-zero rd_i by using non-zero angles and axes that are not aligned with the coord.
  // system axes.
  std::vector<Quaternion> q0s = {
      kqI,
      Quaternion::FromAxisAngle(kvx, ToRads(50_r)),
      Quaternion::FromAxisAngle(Normalize(kv123), ToRads(-20_r))};
  real constexpr kEps = 1.0e-1_r;
  std::vector<Real3> axes = {Normalize(Real3{1_r, kEps, kEps}), Normalize(kv123)};
  std::vector<real> angles = {ToRads(15_r), ToRads(-105_r)};
  std::vector<real> diffs = {ToRads(30_r), ToRads(-135_r)};
  for (auto q0 : q0s) {
    for (auto axis_A : axes) {
      for (auto angle_A : angles) {
        Quaternion qA = Quaternion::FromAxisAngle(axis_A, angle_A);
        for (auto axis_Ar : axes) {
          for (auto angle_Ar : angles) {
            Quaternion qAr = Quaternion::FromAxisAngle(axis_Ar, angle_Ar);
            for (auto axis_d : axes) {
              for (auto angle_d : diffs) {
                Quaternion qd = Quaternion::FromAxisAngle(axis_d, angle_d);
                for (auto axis_diffr : axes) {
                  for (auto angle_diffr : diffs) {
                    Quaternion qdiffr = Quaternion::FromAxisAngle(axis_diffr, angle_diffr);
                    Quaternion qdiff = qdiffr * qd; // qdiff = qdiffr * qd
                    Quaternion qB = Normalize(qA * qdiff); // qB = qA * qdiff = qA * qdiffr * qd
                    Quaternion qBr = Normalize(qAr * qdiffr); // qBr = qAr * qdiffr
                    testFD(qA, qB, qAr, qBr, q0);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

TEST(Constraints, JointRotationTargetJacobianTarget) {
  auto testFD = [](Quaternion const& qA,
                   Quaternion const& qB,
                   Quaternion const& qAr,
                   Quaternion const& qBr,
                   Quaternion const& q0) {
    real constexpr kDelta = 1.0e-2_r;
    real constexpr kTol = 1.0e-2_r;

    auto qr = qAr.GetConjugate() * qBr;

    ColumnVector<real, 3> val;
    RowMatrix<real, 3, 3> Jac;
    EvalJointRotationTargetConstraint(qA, qB, q0, qr, &val, nullptr, &Jac);

    auto JacFD = Jac.Duplicate();
    for (int i = 0; i < 3; ++i) {
      Real3 delta{};
      delta[i] = kDelta;
      auto qrp = Quaternion::FromRotationVector(delta) * qr;
      ColumnVector<real, 3> valp;
      EvalJointRotationTargetConstraint(qA, qB, q0, qrp, &valp, nullptr, nullptr);
      delta[i] = -kDelta;
      auto qrm = Quaternion::FromRotationVector(delta) * qr;
      ColumnVector<real, 3> valm;
      EvalJointRotationTargetConstraint(qA, qB, q0, qrm, &valm, nullptr, nullptr);

      JacFD.Col(i) = SelectClosest(val, valp, valm, Jac.Col(i), kDelta);
    }

    EXPECT_LE((RowMatrix<real, 3, 3>(JacFD - Jac)).Norm(), kTol * JacFD.Norm());
  };

  // Arbritary angles that cover different ranges and combinations
  // Note that finite differences will fail if any rd_i is zero because every component of
  // c = abs(rd) is non-differentiable at rd_i = 0. We ensure all test combinations produce
  // non-zero rd_i by using non-zero angles and axes that are not aligned with the coord.
  // system axes.
  std::vector<Quaternion> q0s = {
      kqI,
      Quaternion::FromAxisAngle(kvx, ToRads(50_r)),
      Quaternion::FromAxisAngle(Normalize(kv123), ToRads(-20_r))};
  real constexpr kEps = 1.0e-1_r;
  std::vector<Real3> axes = {Normalize(Real3{1_r, kEps, kEps}), Normalize(kv123)};
  std::vector<real> angles = {ToRads(15_r), ToRads(-105_r)};
  std::vector<real> diffs = {ToRads(30_r), ToRads(-135_r)};
  for (auto q0 : q0s) {
    for (auto axis_A : axes) {
      for (auto angle_A : angles) {
        Quaternion qA = Quaternion::FromAxisAngle(axis_A, angle_A);
        for (auto axis_Ar : axes) {
          for (auto angle_Ar : angles) {
            Quaternion qAr = Quaternion::FromAxisAngle(axis_Ar, angle_Ar);
            for (auto axis_d : axes) {
              for (auto angle_d : diffs) {
                Quaternion qd = Quaternion::FromAxisAngle(axis_d, angle_d);
                for (auto axis_diffr : axes) {
                  for (auto angle_diffr : diffs) {
                    Quaternion qdiffr = Quaternion::FromAxisAngle(axis_diffr, angle_diffr);
                    Quaternion qdiff = qdiffr * qd; // qdiff = qdiffr * qd
                    Quaternion qB = Normalize(qA * qdiff); // qB = qA * qdiff = qA * qdiffr * qd
                    Quaternion qBr = Normalize(qAr * qdiffr); // qBr = qAr * qdiffr
                    testFD(qA, qB, qAr, qBr, q0);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

/// EvalRigidPositionFixedConstraint ---------------------------------------------
TEST(Constraints, RigidPositionValue) {
  // Define method to evaluate constraint value
  auto testCVal = [](Quaternion const& qs,
                     Real3 const& ts,
                     Real3 const& local,
                     Real3 const& comLocal,
                     Real3 const& target,
                     Real3 const& cVal) {
    real constexpr kTol = 1.0e-2_r;

    TransformRT rbState(qs, ts);

    ColumnVector<real, 3> val;
    EvalRigidPositionFixedConstraint(rbState, local - comLocal, target, &val, nullptr, nullptr);
    for (int i = 0; i < 3; ++i) {
      EXPECT_NEAR_RTOL(cVal[i], val[i], kTol);
    }
  };

  std::vector<real> angles = {
      ToRads(0_r),
      ToRads(-15_r),
      ToRads(90_r),
      ToRads(-120_r),
      ToRads(180_r),
      ToRads(-240_r),
      ToRads(270_r),
      ToRads(-345_r),
      ToRads(360_r),
  };
  std::vector<Real3> axes = {kvx, kvy, kvz, Normalize(kv123)};
  std::vector<Real3> vs = {kv0, kvx, kvy, kvz, kv123};

  for (auto axis : axes) {
    for (auto ts : vs) {
      for (auto target : vs) {
        for (auto local : vs) {
          for (auto comLocal : vs) {
            for (auto angle : angles) {
              Quaternion qs = Quaternion::FromAxisAngle(axis, angle);
              Real3 cval = TransformRT(qs, ts).TransformPoint(local - comLocal) - target;
              testCVal(qs, ts, local, comLocal, target, cval);
            }
          }
        }
      }
    }
  }
}

TEST(Constraints, RigidPositionJacobian) {
  // Define method to evaluate constraint Jacobian
  // NOTE: Because we are using clamped rotations, general finite differences may
  // lead to large errors if the delta triggers the clamping. We get around this
  // issue by checking the derivative estimate with +delta and -delta and picking
  // the one that produces the smallest error.
  auto testFD = [](Quaternion qs, Real3 const& ts, Real3 const& local, Real3 const& target) {
    real constexpr kDelta = 1.0e-2_r;
    real constexpr kTol = 1.0e-2_r;

    TransformRT rbState(qs);

    ColumnVector<real, 3> val;
    RowMatrix<real, 3, 6> Jac;
    EvalRigidPositionFixedConstraint(rbState, local, target, &val, &Jac, nullptr);

    auto JacFD = Jac.Duplicate();
    Real3 t = ts;
    for (int i = 0; i < 3; ++i) {
      t[i] = ts[i] + kDelta;
      rbState.SetTranslation(t);
      EvalRigidPositionFixedConstraint(rbState, local, target, &val, nullptr, nullptr);
      auto valp = val.Duplicate();

      t[i] = ts[i] - kDelta;
      rbState.SetTranslation(t);
      EvalRigidPositionFixedConstraint(rbState, local, target, &val, nullptr, nullptr);
      auto valm = val.Duplicate();
      t[i] = ts[i];

      JacFD.Col(i) = (valp - valm) * (1_r / (2_r * kDelta));
    }
    rbState.SetTranslation(ts);

    for (int i = 0; i < 3; ++i) {
      Real3 delta{};
      delta[i] = kDelta;
      rbState.SetRotation(Quaternion::FromRotationVector(delta) * qs);
      EvalRigidPositionFixedConstraint(rbState, local, target, &val, nullptr, nullptr);
      auto valp = val.Duplicate();

      delta[i] = -kDelta;
      rbState.SetRotation(Quaternion::FromRotationVector(delta) * qs);
      EvalRigidPositionFixedConstraint(rbState, local, target, &val, nullptr, nullptr);
      auto valm = val.Duplicate();

      JacFD.Col(i + 3) = (valp - valm) * (1_r / (2_r * kDelta));
    }
    rbState.SetRotation(qs);

    EXPECT_LE((RowMatrix<real, 3, 6>(JacFD - Jac)).Norm(), kTol * JacFD.Norm());
  };

  std::vector<real> angles = {
      ToRads(15_r),
      ToRads(-90_r),
      ToRads(120_r),
      ToRads(-345_r),
  };
  std::vector<Real3> axes = {kvx, Normalize(kv123)};
  std::vector<Real3> vs = {kv0, kvy, kv123};

  for (auto const& axis : axes) {
    for (auto const& ts : vs) {
      for (auto const& target : vs) {
        for (auto const& local : vs) {
          for (auto const& angle : angles) {
            Quaternion qs = Quaternion::FromAxisAngle(axis, angle);
            testFD(qs, ts, local, target);
          }
        }
      }
    }
  }
}

TEST(Constraints, RigidPositionJacobianTarget) {
  auto testFD = [](Quaternion qs, Real3 const& ts, Real3 const& local, Real3 const& target) {
    real constexpr kDelta = 1.0e-2_r;
    real constexpr kTol = 1.0e-2_r;

    TransformRT rbState(qs, ts);

    ColumnVector<real, 3> val;
    RowMatrix<real, 3, 3> jacTarget;
    EvalRigidPositionFixedConstraint(rbState, local, target, &val, nullptr, &jacTarget);

    auto jacFD = jacTarget.Duplicate();
    Real3 t = target;
    for (int i = 0; i < 3; ++i) {
      t[i] = target[i] + kDelta;
      EvalRigidPositionFixedConstraint(rbState, local, t, &val, nullptr, nullptr);
      auto valp = val.Duplicate();

      t[i] = target[i] - kDelta;
      EvalRigidPositionFixedConstraint(rbState, local, t, &val, nullptr, nullptr);
      auto valm = val.Duplicate();
      t[i] = target[i];

      jacFD.Col(i) = (valp - valm) * (1_r / (2_r * kDelta));
    }

    EXPECT_LE((RowMatrix<real, 3, 3>(jacFD - jacTarget)).Norm(), kTol * jacFD.Norm());
  };

  std::vector<real> angles = {
      ToRads(15_r),
      ToRads(-90_r),
      ToRads(120_r),
      ToRads(-345_r),
  };
  std::vector<Real3> axes = {kvx, Normalize(kv123)};
  std::vector<Real3> vs = {kv0, kvy, kv123};

  for (auto const& axis : axes) {
    for (auto const& ts : vs) {
      for (auto const& target : vs) {
        for (auto const& local : vs) {
          for (auto const& angle : angles) {
            Quaternion qs = Quaternion::FromAxisAngle(axis, angle);
            testFD(qs, ts, local, target);
          }
        }
      }
    }
  }
}

/// EvalRigidPositionToRigidTargetConstraint---------------------------------------------
TEST(Constraints, RigidPositionToRigidTargetValue) {
  Real3 localPos = kvx;

  // Define method to evaluate constraint value
  auto testCVal = [&](Real3 t, Quaternion q, Real3 tTarget, Quaternion qTarget, Real3 cVal) {
    real kTol = 1.0e-3_r;
    ColumnVector<real, 3> val;

    TransformRT rbState(q, t);
    TransformRT rbStateTarget(qTarget, tTarget);
    EvalRigidPositionToRigidTargetConstraint(
        rbState, rbStateTarget, localPos, &val, nullptr, nullptr);

    for (int i = 0; i < 3; ++i) {
      EXPECT_NEAR_RTOL(cVal[i], val[i], kTol);
    }
  };

  std::vector<Real3> vs = {kv0, kvx, kvy, kvz, kv123};
  std::vector<Quaternion> qs = {
      kqI,
      quatx(ToRads(90_r)),
      quatx(ToRads(180_r)),
      quatx(ToRads(-90_r)),
      quatx(ToRads(-360_r)),
      quaty(ToRads(90_r)),
      quaty(ToRads(180_r)),
      quaty(ToRads(-90_r)),
      quaty(ToRads(-360_r)),
      quatz(ToRads(90_r)),
      quatz(ToRads(180_r)),
      quatz(ToRads(-90_r)),
      quatz(ToRads(-360_r)),
  };

  for (auto t : vs) {
    for (auto q : qs) {
      for (auto tTarget : vs) {
        for (auto qTarget : qs) {
          testCVal(
              t,
              q,
              tTarget,
              qTarget,
              TransformRT(q, t).TransformPoint(localPos) -
                  TransformRT(qTarget, tTarget).TransformPoint(localPos));
        }
      }
    }
  }
}

TEST(Constraints, RigidPositionToRigidTargetJacobian) {
  Real3 localPos = kvx;

  // Define method to evaluate constraint Jacobian
  auto testFD = [&](Real3 t, Quaternion q, Real3 tTarget, Quaternion qTarget) {
    real kDelta = 1e-2_r;
    real kTol = 3e-2_r;
    ColumnVector<real, 3> val;
    RowMatrix<real, 3, 6> Jac;

    TransformRT rbState(q, t);
    TransformRT rbStateTarget(qTarget, tTarget);
    EvalRigidPositionToRigidTargetConstraint(rbState, rbStateTarget, localPos, &val, &Jac, nullptr);

    auto JacFD = Jac.Duplicate();

    for (int i = 0; i < 3; ++i) { // dC/dx
      Real3 x = t;
      x[i] += kDelta;
      rbState.SetTranslation(x);
      EvalRigidPositionToRigidTargetConstraint(
          rbState, rbStateTarget, localPos, &val, nullptr, nullptr);
      auto valp = val.Duplicate();

      x[i] -= 2_r * kDelta;
      rbState.SetTranslation(x);
      EvalRigidPositionToRigidTargetConstraint(
          rbState, rbStateTarget, localPos, &val, nullptr, nullptr);
      auto valm = val.Duplicate();

      JacFD.Col(i) = (valp - valm) * (1_r / (2_r * kDelta));
    }
    rbState.SetTranslation(t);
    for (int i = 0; i < 3; ++i) { // dC/dR
      Real3 delta{};
      delta[i] = kDelta;
      rbState.SetRotation(Quaternion::FromRotationVector(delta) * q);
      EvalRigidPositionToRigidTargetConstraint(
          rbState, rbStateTarget, localPos, &val, nullptr, nullptr);
      auto valp = val.Duplicate();

      delta[i] = -kDelta;
      rbState.SetRotation(Quaternion::FromRotationVector(delta) * q);
      EvalRigidPositionToRigidTargetConstraint(
          rbState, rbStateTarget, localPos, &val, nullptr, nullptr);
      auto valm = val.Duplicate();

      JacFD.Col(i + 3) = (valp - valm) * (1_r / (2_r * kDelta));
    }
    rbState.SetRotation(q);
    EXPECT_LE((RowMatrix<real, 3, 6>(JacFD - Jac)).Norm(), kTol * JacFD.Norm());
  };

  std::vector<Real3> vs = {kv0, kvy + kvz, kv123};
  std::vector<real> angles = {ToRads(0_r), ToRads(-15_r), ToRads(179_r), ToRads(359_r)};
  std::vector<Real3> axes = {kvx, Normalize(kvy + kvz), Normalize(kv123)};
  for (auto t : vs) {
    for (auto angle : angles) {
      for (auto axis : axes) {
        Quaternion q = Quaternion::FromAxisAngle(axis, angle);
        for (auto tTarget : vs) {
          for (auto angleTarget : angles) {
            for (auto axisTarget : axes) {
              Quaternion qTarget = Quaternion::FromAxisAngle(axisTarget, angleTarget);
              testFD(t, q, tTarget, qTarget);
            }
          }
        }
      }
    }
  }
}

TEST(Constraints, RigidPositionToRigidTargetJacobianTarget) {
  Real3 localPos = kvx;

  auto testFD = [&](Real3 t, Quaternion q, Real3 tTarget, Quaternion qTarget) {
    real kDelta = 1e-2_r;
    real kTol = 3e-2_r;
    ColumnVector<real, 3> val;
    RowMatrix<real, 3, 6> jacTarget;

    TransformRT rbState(q, t);
    TransformRT rbStateTarget(qTarget, tTarget);
    EvalRigidPositionToRigidTargetConstraint(
        rbState, rbStateTarget, localPos, &val, nullptr, &jacTarget);

    auto JacFD = jacTarget.Duplicate();

    for (int i = 0; i < 3; ++i) { // dC/dxTarget
      Real3 x = tTarget;
      x[i] += kDelta;
      rbStateTarget.SetTranslation(x);
      EvalRigidPositionToRigidTargetConstraint(
          rbState, rbStateTarget, localPos, &val, nullptr, nullptr);
      auto valp = val.Duplicate();

      x[i] -= 2_r * kDelta;
      rbStateTarget.SetTranslation(x);
      EvalRigidPositionToRigidTargetConstraint(
          rbState, rbStateTarget, localPos, &val, nullptr, nullptr);
      auto valm = val.Duplicate();

      JacFD.Col(i) = (valp - valm) * (1_r / (2_r * kDelta));
    }
    rbStateTarget.SetTranslation(tTarget);
    for (int i = 0; i < 3; ++i) { // dC/dRTarget
      Real3 delta{};
      delta[i] = kDelta;
      rbStateTarget.SetRotation(Quaternion::FromRotationVector(delta) * qTarget);
      EvalRigidPositionToRigidTargetConstraint(
          rbState, rbStateTarget, localPos, &val, nullptr, nullptr);
      auto valp = val.Duplicate();

      delta[i] = -kDelta;
      rbStateTarget.SetRotation(Quaternion::FromRotationVector(delta) * qTarget);
      EvalRigidPositionToRigidTargetConstraint(
          rbState, rbStateTarget, localPos, &val, nullptr, nullptr);
      auto valm = val.Duplicate();

      JacFD.Col(i + 3) = (valp - valm) * (1_r / (2_r * kDelta));
    }
    rbStateTarget.SetRotation(qTarget);
    EXPECT_LE((RowMatrix<real, 3, 6>(JacFD - jacTarget)).Norm(), kTol * JacFD.Norm());
  };

  std::vector<Real3> vs = {kv0, kvy + kvz, kv123};
  std::vector<real> angles = {ToRads(0_r), ToRads(-15_r), ToRads(179_r), ToRads(359_r)};
  std::vector<Real3> axes = {kvx, Normalize(kvy + kvz), Normalize(kv123)};
  for (auto t : vs) {
    for (auto angle : angles) {
      for (auto axis : axes) {
        Quaternion q = Quaternion::FromAxisAngle(axis, angle);
        for (auto tTarget : vs) {
          for (auto angleTarget : angles) {
            for (auto axisTarget : axes) {
              Quaternion qTarget = Quaternion::FromAxisAngle(axisTarget, angleTarget);
              testFD(t, q, tTarget, qTarget);
            }
          }
        }
      }
    }
  }
}

/// EvalDeformableNodeToRigidConstraint ---------------------------------------------
TEST(Constraints, DeformableNodeToRigidValue) {
  // Define method to evaluate constraint
  auto testCVal = [](TransformRT const& rigidTx,
                     Real3 const& rigidRest,
                     TransformRT const& deformableTx,
                     Real3 const& deformableRest,
                     Real3 const& deformableDisp,
                     Real3 const& cVal) {
    // Compute constraint value
    ColumnVector<real, 3> val;
    EvalDeformableNodeToRigidConstraint(
        rigidTx, deformableTx, deformableDisp + deformableRest, rigidRest, &val, nullptr);
    real constexpr kTol = 1.0e-3_r;
    for (int i = 0; i < 3; ++i) {
      EXPECT_NEAR_RTOL(cVal[i], val[i], kTol);
    }
  };

  auto computeCVal =
      [](TransformRT const& txRigid, // Method to extract displacement from given context
         Real3 const& rr,
         TransformRT const& txDeformable,
         Real3 X,
         Real3 u) { return txRigid.TransformPoint(rr) - txDeformable.TransformPoint(X + u); };

  std::vector<TransformRT> txsRigid = {
      TransformRT(kqI, kv0),
      TransformRT(Quaternion::FromAxisAngle(kvx, ToRads(30_r)), kv0),
      TransformRT(Quaternion::FromAxisAngle(kvz, ToRads(120_r)), kvy),
      TransformRT(Quaternion::FromAxisAngle(Normalize(kv123), ToRads(-20_r)), kv123),
  };
  std::vector<TransformRT> txsDeformable = {
      TransformRT(kqI, kv0),
      TransformRT(Quaternion::FromAxisAngle(kvy, ToRads(-30_r)), kv0),
      TransformRT(Quaternion::FromAxisAngle(kvz, ToRads(120_r)), -kvy),
      TransformRT(Quaternion::FromAxisAngle(Normalize(kv123), ToRads(60_r)), kv123),
  };
  std::vector<Real3> vs = {kv0, kvx, kvy, kvz, kv123};

  for (auto txRigid : txsRigid) {
    for (auto rRigid : vs) {
      for (auto txDeformable : txsDeformable) {
        for (auto deformableRest : vs) {
          for (auto deformableDisp : vs) {
            Real3 cval = computeCVal(txRigid, rRigid, txDeformable, deformableRest, deformableDisp);
            testCVal(txRigid, rRigid, txDeformable, deformableRest, deformableDisp, cval);
          }
        }
      }
    }
  }
}

TEST(Constraints, DeformableNodeToRigidJacobian) {
  // Define method to evaluate Jacobian through finite differences
  auto testFD = [](TransformRT const& rigidTx,
                   Real3 const& rigidRest,
                   TransformRT const& deformableTx,
                   Real3 const& deformableRest,
                   Real3 const& deformableDisp) {
    // Update actor-related parameters according to test input
    TransformRT rbState = rigidTx;
    auto deformableLocal = deformableDisp + deformableRest;

    ColumnVector<real, 3> val;
    RowMatrix<real, 3, 9> Jac;
    EvalDeformableNodeToRigidConstraint(
        rbState, deformableTx, deformableLocal, rigidRest, &val, &Jac);

    auto JacFD = Jac.Duplicate();

    real constexpr kDelta = 1.0e-2_r;
    real constexpr kTol = 1.0e-2_r;

    // Compute FD components for rigid body dofs
    Real3 x0 = rbState.GetTranslation();
    Quaternion q0 = rbState.GetRotation();
    for (int i = 0; i < 3; ++i) { // Position
      Real3 xA = x0;
      xA[i] += kDelta;
      rbState.SetTranslation(xA);
      EvalDeformableNodeToRigidConstraint(
          rbState, deformableTx, deformableLocal, rigidRest, &val, nullptr);
      auto valp = val.Duplicate();

      xA[i] -= 2_r * kDelta;
      rbState.SetTranslation(xA);
      EvalDeformableNodeToRigidConstraint(
          rbState, deformableTx, deformableLocal, rigidRest, &val, nullptr);
      auto valm = val.Duplicate();

      JacFD.Col(i) = (valp - valm) * (1_r / (2_r * kDelta));
    }
    rbState.SetTranslation(x0);

    for (int i = 0; i < 3; ++i) { // Orientation
      Real3 delta{};
      delta[i] = kDelta;
      rbState.SetRotation(Quaternion::FromRotationVector(delta) * q0);
      EvalDeformableNodeToRigidConstraint(
          rbState, deformableTx, deformableLocal, rigidRest, &val, nullptr);
      auto valp = val.Duplicate();

      delta[i] = -kDelta;
      rbState.SetRotation(Quaternion::FromRotationVector(delta) * q0);
      EvalDeformableNodeToRigidConstraint(
          rbState, deformableTx, deformableLocal, rigidRest, &val, nullptr);
      auto valm = val.Duplicate();

      JacFD.Col(i + 3) = (valp - valm) * (1_r / (2_r * kDelta));
    }
    rbState.SetRotation(q0);

    // Compute FD components for deformable node dofs
    Real3 disp = deformableDisp;
    for (int i = 0; i < 3; ++i) {
      real v0 = disp[i];
      disp[i] += kDelta;
      deformableLocal = disp + deformableRest;
      EvalDeformableNodeToRigidConstraint(
          rbState, deformableTx, deformableLocal, rigidRest, &val, nullptr);
      auto valp = val.Duplicate();

      disp[i] -= 2_r * kDelta;
      deformableLocal = disp + deformableRest;
      EvalDeformableNodeToRigidConstraint(
          rbState, deformableTx, deformableLocal, rigidRest, &val, nullptr);
      auto valm = val.Duplicate();
      disp[i] = v0;

      JacFD.Col(i + 6) = (valp - valm) * (1_r / (2_r * kDelta));
    }

    EXPECT_LE((RowMatrix<real, 3, 9>(JacFD - Jac)).Norm(), kTol * JacFD.Norm());
  };

  std::vector<TransformRT> txsRigid = {
      TransformRT(kqI, kv0),
      TransformRT(Quaternion::FromAxisAngle(kvx, ToRads(30_r)), kv0),
      TransformRT(Quaternion::FromAxisAngle(kvz, ToRads(120_r)), kvy),
      TransformRT(Quaternion::FromAxisAngle(Normalize(kv123), ToRads(-20_r)), kv123),
  };
  std::vector<TransformRT> txsDeformable = {
      TransformRT(kqI, kv0),
      TransformRT(Quaternion::FromAxisAngle(kvy, ToRads(-30_r)), kv0),
      TransformRT(Quaternion::FromAxisAngle(kvz, ToRads(120_r)), -kvy),
      TransformRT(Quaternion::FromAxisAngle(Normalize(kv123), ToRads(60_r)), kv123),
  };
  std::vector<Real3> vs = {kv0, kvx, kvy, kvz, kv123};

  for (auto txRigid : txsRigid) {
    for (auto rRigid : vs) {
      for (auto txDeformable : txsDeformable) {
        for (auto deformableRest : vs) {
          for (auto deformableDisp : vs) {
            testFD(txRigid, rRigid, txDeformable, deformableRest, deformableDisp);
          }
        }
      }
    }
  }
}

/// EvalSingleDofTargetConstraint ---------------------------------------------
TEST(Constraints, SingleDofTargetValue) {
  // Define method to evaluate constraint
  auto testCVal = [](Span<real const> targetValues, Span<real const> dofValues) {
    ColumnVector<real> eVal = ColumnVectorView<real const>(dofValues.data(), dofValues.size()) -
        ColumnVectorView<real const>(targetValues.data(), targetValues.size());

    ColumnVector<real> cVal(isize(dofValues)); // Evaluated constraint value
    for (int i = 0; i < isize(dofValues); ++i) {
      EvalSingleDofTargetConstraint(dofValues[i], targetValues[i], &cVal(i), nullptr, nullptr);
    }
    ColumnVector<real> err = cVal - eVal;
    real constexpr kTol = 1.0e-3_r;
    EXPECT_LE(err.Norm(), kTol);
  };

  // Test base cases with 3-dimensional vectors
  std::vector<Real3> vs = {kv0, kvx, kvy, kvz, kv123, -kv123};
  for (auto targetValues : vs) {
    for (auto dofValues : vs) {
      testCVal(targetValues, dofValues);
    }
  }

  // Test cases with variable number of dofs
  struct TestData {
    std::vector<real> targetValues;
    std::vector<real> dofValues;
  };

  std::vector<TestData> testConfigs = {
      {{-1_r}, {0.5_r}},
      {{-1_r, 0_r}, {-1.5_r, 0.5_r}},
      {{-1_r, -1_r, 0_r}, {-2_r, 0_r, 3_r}},
      {{0_r, 0_r, 0_r}, {-1_r, 1_r, 4_r}}};
  for (auto tc : testConfigs) {
    testCVal(tc.targetValues, tc.dofValues);
  }
}

TEST(Constraints, SingleDofTargetJacobian) {
  // Define method to evaluate Jacobian through finite differences
  auto testFD = [](Span<real const> targetValues, Span<real const> dofValues) {
    auto eval = [](Span<real const> targetValues,
                   Span<real const> dofValues,
                   ColumnVector<real>* val,
                   RowMatrix<real>* jac) {
      if (jac) {
        jac->SetZero();
      }
      for (int i = 0; i < isize(dofValues); ++i) {
        auto* jacPtr = jac ? &(*jac)(i, i) : nullptr;
        EvalSingleDofTargetConstraint(dofValues[i], targetValues[i], &(*val)(i), jacPtr, nullptr);
      }
    };

    ColumnVector<real> val(isize(dofValues));
    RowMatrix<real> jac(isize(dofValues), isize(dofValues));
    eval(targetValues, dofValues, &val, &jac);

    auto jacFD = jac.Duplicate();
    real constexpr kDelta = 1.0e-3_r;
    real constexpr kTol = 1.0e-3_r;
    auto dofValuesFD = AsConstView(dofValues).Duplicate(); // Copy to modify locally
    for (int i = 0; i < isize(dofValues); ++i) {
      real v0 = dofValuesFD[i];
      real dx = std::max(kDelta, std::abs(kDelta * dofValuesFD[i]));
      dofValuesFD[i] += dx;
      eval(targetValues, dofValuesFD, &val, nullptr);
      auto valp = val.Duplicate();

      dofValuesFD[i] -= 2_r * dx;
      eval(targetValues, dofValuesFD, &val, nullptr);
      auto valm = val.Duplicate();
      dofValuesFD[i] = v0;

      jacFD.Col(i) = (valp - valm) * (1_r / (2_r * dx));
    }
    EXPECT_LE((RowMatrix<real>(jacFD - jac)).Norm(), kTol * jacFD.Norm());
  };

  std::vector<std::pair<std::vector<real>, std::vector<real>>> pairs = {
      {{0_r}, {1_r}},
      {{0_r, 1_r}, {0_r, 0_r}},
      {{0_r, 1_r, 2_r}, {0_r, 0_r, 2_r}},
      {{1_r, 2_r, 3_r}, {0_r, 2_r, 5_r}}};
  for (auto const& p : pairs) {
    auto const& targetValues = p.first;
    auto const& dofValues = p.second;
    testFD(targetValues, dofValues);
  }
}

TEST(Constraints, SingleDofTargetJacobianTarget) {
  auto testFD = [](Span<real const> targetValues, Span<real const> dofValues) {
    auto eval = [](Span<real const> targetValues,
                   Span<real const> dofValues,
                   ColumnVector<real>* val,
                   RowMatrix<real>* jac) {
      if (jac) {
        jac->SetZero();
      }
      for (int i = 0; i < isize(dofValues); ++i) {
        auto* jacPtr = jac ? &(*jac)(i, i) : nullptr;
        EvalSingleDofTargetConstraint(dofValues[i], targetValues[i], &(*val)(i), nullptr, jacPtr);
      }
    };

    ColumnVector<real> val(isize(dofValues));
    RowMatrix<real> jac(isize(dofValues), isize(dofValues));
    eval(targetValues, dofValues, &val, &jac);

    auto jacFD = jac.Duplicate();
    real constexpr kDelta = 1.0e-3_r;
    real constexpr kTol = 1.0e-3_r;
    auto targetValuesFD = AsConstView(targetValues).Duplicate(); // Copy to modify locally
    for (int i = 0; i < isize(dofValues); ++i) {
      real v0 = targetValuesFD[i];
      real dx = std::max(kDelta, std::abs(kDelta * targetValuesFD[i]));
      targetValuesFD[i] += dx;
      eval(targetValuesFD, dofValues, &val, nullptr);
      auto valp = val.Duplicate();

      targetValuesFD[i] -= 2_r * dx;
      eval(targetValuesFD, dofValues, &val, nullptr);
      auto valm = val.Duplicate();
      targetValuesFD[i] = v0;

      jacFD.Col(i) = (valp - valm) * (1_r / (2_r * dx));
    }
    EXPECT_LE((RowMatrix<real>(jacFD - jac)).Norm(), kTol * jacFD.Norm());
  };

  std::vector<std::pair<std::vector<real>, std::vector<real>>> pairs = {
      {{0_r}, {1_r}},
      {{0_r, 1_r}, {0_r, 0_r}},
      {{0_r, 1_r, 2_r}, {0_r, 0_r, 2_r}},
      {{1_r, 2_r, 3_r}, {0_r, 2_r, 5_r}}};
  for (auto const& p : pairs) {
    auto const& targetValues = p.first;
    auto const& dofValues = p.second;
    testFD(targetValues, dofValues);
  }
}

/// EvalSingleDofRangeConstraint ---------------------------------------------
TEST(Constraints, SingleDofRangeValue) {
  // Define method to evaluate constraint
  auto testCVal =
      [](Span<real const> minValues, Span<real const> maxValues, Span<real const> dofValues) {
        auto eVal = ColumnVector<real>::Zero(isize(dofValues));
        for (int i = 0; i < isize(dofValues); ++i) {
          if (dofValues[i] < minValues[i]) {
            eVal[i] = minValues[i] - dofValues[i];
          } else if (dofValues[i] > maxValues[i]) {
            eVal[i] = dofValues[i] - maxValues[i];
          } else {
            eVal[i] = 0_r;
          }
        }
        ColumnVector<real> cVal(isize(dofValues)); // Evaluated constraint value
        for (int i = 0; i < isize(dofValues); ++i) {
          bool isActive{};
          EvalSingleDofRangeConstraint(
              dofValues[i], minValues[i], maxValues[i], &cVal(i), {}, isActive);
        }

        real constexpr kTol = 1.0e-3_r;
        for (int i = 0; i < isize(dofValues); ++i) {
          EXPECT_NEAR_RTOL(cVal[i], eVal[i], kTol);
        }
      };

  // Test base cases with 3-dimensional vectors
  std::vector<Real3> vs = {kv0, kvx, kvy, kvz, kv123, -kv123};
  for (auto v0 : vs) {
    for (auto v1 : vs) {
      for (auto dofValues : vs) {
        auto minValues = Min(v0, v1);
        auto maxValues = Max(v0, v1);
        testCVal(minValues, maxValues, dofValues);
      }
    }
  }

  // Test cases with variable number of dofs
  struct TestData {
    std::vector<real> minValues;
    std::vector<real> maxValues;
    std::vector<real> dofValues;
  };

  std::vector<TestData> testConfigs = {
      {{-1_r}, {1_r}, {0.5_r}},
      {{-1_r, 0_r}, {1_r, 1_r}, {-1.5_r, 0.5_r}},
      {{-1_r, -1_r, 0_r}, {1_r, 1_r, 2_r}, {-2_r, 0_r, 3_r}},
      {{0_r, 0_r, 0_r}, {1_r, 2_r, 3_r}, {-1_r, 1_r, 4_r}}};
  for (auto tc : testConfigs) {
    testCVal(tc.minValues, tc.maxValues, tc.dofValues);
  }
}

TEST(Constraints, SingleDofRangeJacobian) {
  real delta = 0.1_r;
  int ndim = 3;
  std::vector<std::pair<real, real>> boundPairs = {{-1_r, 1_r}, {0_r, 1_r}, {-1_r, 0_r}};
  std::vector<real> minValues(ndim);
  std::vector<real> maxValues(ndim);
  ColumnVector<real> dofValues(ndim);
  for (auto bp : boundPairs) {
    // Define method to evaluate Jacobian through finite differences
    auto testFD = [&]() {
      auto eval = [](Span<real const> dofs,
                     Span<real const> min,
                     Span<real const> max,
                     ColumnVector<real>* val,
                     RowMatrix<real>* jac) {
        if (jac) {
          jac->SetZero();
        }
        for (int i = 0; i < isize(dofs); ++i) {
          auto* jacPtr = jac ? &(*jac)(i, i) : nullptr;
          bool isActive{};
          EvalSingleDofRangeConstraint(dofs[i], min[i], max[i], &(*val)(i), jacPtr, isActive);
        }
      };

      ColumnVector<real> val(ndim);
      RowMatrix<real> jac(ndim, ndim);
      eval(dofValues, minValues, maxValues, &val, &jac);

      RowMatrix<real> jacFD(ndim, ndim);
      real constexpr kDelta = 1.0e-3_r;
      real constexpr kTol = 1.0e-3_r;
      for (int i = 0; i < ndim; ++i) {
        ColumnVector<real> dofValuesFD = dofValues.Duplicate();
        dofValuesFD[i] += kDelta;
        eval(dofValuesFD, minValues, maxValues, &val, nullptr);
        ColumnVector<real> valp = val;

        dofValuesFD[i] -= 2_r * kDelta;
        eval(dofValuesFD, minValues, maxValues, &val, nullptr);
        ColumnVector<real> valm = val;

        jacFD.Col(i) = (valp - valm) * (1_r / (2_r * kDelta));
      }
      EXPECT_LE((RowMatrix<real>(jacFD - jac)).Norm(), kTol * jacFD.Norm());
    };

    // Test individual coordinates
    for (int i = 0; i < ndim; ++i) {
      std::fill(minValues.begin(), minValues.end(), 0_r);
      std::fill(maxValues.begin(), maxValues.end(), 0_r);
      minValues[i] = bp.first;
      maxValues[i] = bp.second;

      dofValues.SetZero();
      dofValues[i] = minValues[i] - delta;
      testFD(); // value < minValue
      dofValues[i] = 0.5_r * (maxValues[i] + minValues[i]);
      testFD(); // minValue <= value <= maxValue
      dofValues[i] = maxValues[i] + delta;
      testFD(); // value > maxValue
    }

    // Test all coordinates
    std::fill(minValues.begin(), minValues.end(), bp.first);
    std::fill(maxValues.begin(), maxValues.end(), bp.second);
    for (auto i = 0; i < ndim; ++i) {
      if (i % 3 == 0) {
        dofValues[i] = bp.first - delta;
      } else if (i % 3 == 1) {
        dofValues[i] = 0.5_r * (bp.second + bp.first);
      } else if (i % 3 == 2) {
        dofValues[i] = bp.second + delta;
      }
    }
    testFD();
  }
}

/// EvalRotationRangeConstraint ---------------------------------------------
TEST(Constraints, RotationRangeJacobian) {
  real delta = 0.1_r;
  std::array<std::pair<real, real>, 3> boundPairs = {{{-1_r, 1_r}, {0_r, 1_r}, {-1_r, 0_r}}};
  Real3 minValues;
  Real3 maxValues;
  Real3 dofValues;
  for (auto bp : boundPairs) {
    // Define method to evaluate Jacobian through finite differences
    auto testFD = [&]() {
      Real3 rot0 = dofValues;
      Quaternion q0 = Quaternion::FromRotationVector(rot0);

      ColumnVector<real, 3> val;
      RowMatrix<real, 3, 3> jac;
      bool isActive{};
      EvalRotationRangeConstraint(rot0, minValues, maxValues, &val, &jac, isActive);

      auto jacFD = jac.Duplicate();
      real constexpr kDelta = 1.0e-3_r;
      real constexpr kTol = 1.0e-3_r;
      for (int i = 0; i < 3; ++i) {
        Real3 delta3{};
        delta3[i] = kDelta;
        auto rot = (Quaternion::FromRotationVector(delta3) * q0).ToRotationVector();
        EvalRotationRangeConstraint(rot, minValues, maxValues, &val, nullptr, isActive);
        auto valp = val.Duplicate();

        delta3[i] = -kDelta;
        rot = (Quaternion::FromRotationVector(delta3) * q0).ToRotationVector();
        EvalRotationRangeConstraint(rot, minValues, maxValues, &val, nullptr, isActive);
        auto valm = val.Duplicate();

        jacFD.Col(i) = (valp - valm) * (1_r / (2_r * kDelta));
      }
      EXPECT_LE((RowMatrix<real>(jacFD - jac)).Norm(), kTol * jacFD.Norm());
    };

    // Test individual coordinates
    for (int i = 0; i < 3; ++i) {
      std::fill(minValues.begin(), minValues.end(), 0_r);
      std::fill(maxValues.begin(), maxValues.end(), 0_r);
      minValues[i] = bp.first;
      maxValues[i] = bp.second;

      dofValues = {};
      dofValues[i] = minValues[i] - delta;
      testFD(); // value < minValue
      dofValues[i] = 0.5_r * (maxValues[i] + minValues[i]);
      testFD(); // minValue <= value <= maxValue
      dofValues[i] = maxValues[i] + delta;
      testFD(); // value > maxValue
    }

    // Test all coordinates
    std::fill(minValues.begin(), minValues.end(), bp.first);
    std::fill(maxValues.begin(), maxValues.end(), bp.second);
    for (auto i = 0; i < 3; ++i) {
      if (i % 3 == 0) {
        dofValues[i] = bp.first - delta;
      } else if (i % 3 == 1) {
        dofValues[i] = 0.5_r * (bp.second + bp.first);
      } else if (i % 3 == 2) {
        dofValues[i] = bp.second + delta;
      }
    }
    testFD();
  }
}

/// EvalRodElementRotationToRigidConstraint ---------------------------------------------
TEST(Constraints, RodElementRotationToRigidValue) {
  // Define method to evaluate constraint value
  auto const testCVal = [](Quaternion const& qA,
                           Quaternion const& qr,
                           TransformRT const& worldFromLocalRod,
                           Vec4r const& X0Rod,
                           Vec4r const& X1Rod,
                           Vec4r const& frameAxisRod,
                           Span<real const> rodElementDofsB,
                           Quaternion const& q0,
                           Real3 const& cVal) {
    ColumnVector<real, 3> val;
    EvalRodElementRotationToRigidConstraint(
        qA,
        worldFromLocalRod,
        X0Rod,
        X1Rod,
        frameAxisRod,
        rodElementDofsB,
        q0,
        qr,
        &val,
        nullptr,
        nullptr);

    real constexpr kTol = 1.0e-2_r;
    for (int i = 0; i < 3; ++i) {
      EXPECT_NEAR_RTOL(cVal[i], val[i], kTol);
    }
  };

  // Setup test data for rod element
  // Simple rod element with nodes at (0,0,0) and (1,0,0)
  Vec4r const X0 = ToSimd(Real3{0_r, 0_r, 0_r});
  Vec4r const X1 = ToSimd(Real3{1_r, 0_r, 0_r});
  Vec4r const frameAxis = ToSimd(Real3{0_r, 1_r, 0_r}); // Y-axis

  // Given rod element DoF configurations (treating them as test data)
  DynamicArray<DynamicArray<real>> const rodDofsConfigs = {
      {0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r},
      {0_r, 0_r, 0_r, 0.1_r, 0_r, 0_r, 0_r, 0.1_r},
      {0_r, 0_r, 0_r, 0.2_r, 0_r, 0_r, 0_r, 0.3_r},
      {0.05_r, 0_r, 0_r, 0.15_r, 0_r, 0.05_r, 0_r, 0.25_r},
      {0_r, 0.05_r, 0_r, -0.1_r, 0_r, 0_r, 0.05_r, -0.2_r}};

  // Arbitrary angles that cover different ranges and combinations
  DynamicArray<Quaternion> const q0s = {
      Quaternion::FromAxisAngle(kvx, ToRads(50_r)),
      Quaternion::FromAxisAngle(Normalize(kv123), ToRads(-20_r))};
  DynamicArray<Real3> const axes = {kvx, kvy, kvz, Normalize(kv123)};
  DynamicArray<real> const angles = {ToRads(15_r), ToRads(-105_r)};

  DynamicArray<TransformRT> const transforms = {
      TransformRT(kqI, kv0), TransformRT(quatx(ToRads(30_r)), kvx)};

  for (auto const& q0 : q0s) {
    for (auto const& axis_A : axes) {
      for (auto const& angle_A : angles) {
        Quaternion const qA = Quaternion::FromAxisAngle(axis_A, angle_A);
        for (auto const& axis_Ar : axes) {
          for (auto const& angle_Ar : angles) {
            Quaternion const qAr = Quaternion::FromAxisAngle(axis_Ar, angle_Ar);
            for (auto const& rodDofsB : rodDofsConfigs) {
              for (auto const& rodDofsBr : rodDofsConfigs) {
                for (auto wflRod : transforms) {
                  // Compute qB from rodDofsB
                  VMatrix3x3r rotBMatLocal MOCHI_NO_INIT;
                  fem::ComputeRodElementRotationLocal(
                      X0, X1, frameAxis, rodDofsB, rotBMatLocal, nullptr);
                  Quaternion qB =
                      Normalize(wflRod.GetRotation() * QuaternionFromMatrix(rotBMatLocal));

                  // Compute qBr from rodDofsBr
                  VMatrix3x3r rotBrMatLocal MOCHI_NO_INIT;
                  fem::ComputeRodElementRotationLocal(
                      X0, X1, frameAxis, rodDofsBr, rotBrMatLocal, nullptr);
                  Quaternion const qBr =
                      Normalize(wflRod.GetRotation() * QuaternionFromMatrix(rotBrMatLocal));

                  // Compute expected constraint value
                  // qr is the target relative rotation
                  Quaternion const qr = qAr.GetConjugate() * qBr;
                  // qd is the actual rotation error: qd = (qA * qr * q0)^-1 * qB * q0
                  Quaternion const qd = (qA * qr * q0).GetConjugate() * qB * q0;
                  Real3 const rd = Normalize(qd).ToRotationVector();

                  testCVal(qA, qr, wflRod, X0, X1, frameAxis, rodDofsB, q0, rd);
                }
              }
            }
          }
        }
      }
    }
  }
}

TEST(Constraints, RodElementRotationToRigidJacobian) {
  // Define method to evaluate constraint Jacobian
  auto testFD = [](Quaternion const& rotRigid,
                   TransformRT const& worldFromLocalRod,
                   Vec4r const& X0Rod,
                   Vec4r const& X1Rod,
                   Vec4r const& frameAxisRod,
                   DynamicArray<real> const& rodElementDofs,
                   Quaternion const& refFrame,
                   Quaternion const& targetRot) {
    real constexpr kDelta = 1.0e-2_r;
    real constexpr kTol = 1.0e-2_r;
    ColumnVector<real, 3> val;
    RowMatrix<real, 3, 11> Jac;

    EvalRodElementRotationToRigidConstraint(
        rotRigid,
        worldFromLocalRod,
        X0Rod,
        X1Rod,
        frameAxisRod,
        rodElementDofs,
        refFrame,
        targetRot,
        &val,
        &Jac,
        nullptr);

    auto const val0 = val.Duplicate();
    auto JacFD = Jac.Duplicate();

    // Test derivative w.r.t. rigid rotation (first 3 columns)
    for (int i = 0; i < 3; ++i) {
      Real3 delta{};
      delta[i] = kDelta;
      Quaternion const rotRigidP = Quaternion::FromRotationVector(delta) * rotRigid;
      EvalRodElementRotationToRigidConstraint(
          rotRigidP,
          worldFromLocalRod,
          X0Rod,
          X1Rod,
          frameAxisRod,
          rodElementDofs,
          refFrame,
          targetRot,
          &val,
          nullptr,
          nullptr);
      auto valp = val.Duplicate();

      delta[i] = -kDelta;
      Quaternion const rotRigidM = Quaternion::FromRotationVector(delta) * rotRigid;
      EvalRodElementRotationToRigidConstraint(
          rotRigidM,
          worldFromLocalRod,
          X0Rod,
          X1Rod,
          frameAxisRod,
          rodElementDofs,
          refFrame,
          targetRot,
          &val,
          nullptr,
          nullptr);
      auto const valm = val.Duplicate();

      JacFD.Col(i) = SelectClosest(val0, valp, valm, Jac.Col(i), kDelta);
    }

    // Test derivative w.r.t. rod element DOFs (columns 3-10, i.e., 8 DOFs)
    for (int i = 0; i < 8; ++i) {
      DynamicArray<real> rodDofsP = rodElementDofs;
      rodDofsP[i] += kDelta;
      DynamicArray<real> pertP(8, 0_r);
      pertP[i] = kDelta;
      Vec4r const axisPertP = test::ComputePerturbedFrameAxis(
          X0Rod, X1Rod, MakeConstSpan(rodElementDofs), MakeConstSpan(pertP), frameAxisRod);
      EvalRodElementRotationToRigidConstraint(
          rotRigid,
          worldFromLocalRod,
          X0Rod,
          X1Rod,
          axisPertP,
          rodDofsP,
          refFrame,
          targetRot,
          &val,
          nullptr,
          nullptr);
      auto const valp = val.Duplicate();

      DynamicArray<real> rodDofsM = rodElementDofs;
      rodDofsM[i] -= kDelta;
      DynamicArray<real> pertM(8, 0_r);
      pertM[i] = -kDelta;
      Vec4r const axisPertM = test::ComputePerturbedFrameAxis(
          X0Rod, X1Rod, MakeConstSpan(rodElementDofs), MakeConstSpan(pertM), frameAxisRod);
      EvalRodElementRotationToRigidConstraint(
          rotRigid,
          worldFromLocalRod,
          X0Rod,
          X1Rod,
          axisPertM,
          rodDofsM,
          refFrame,
          targetRot,
          &val,
          nullptr,
          nullptr);
      auto const valm = val.Duplicate();

      JacFD.Col(i + 3) = SelectClosest(val0, valp, valm, Jac.Col(i + 3), kDelta);
    }

    EXPECT_LE((RowMatrix<real, 3, 11>(JacFD - Jac)).Norm(), kTol * JacFD.Norm());
  };

  // Setup test data for rod element
  Vec4r const X0 = ToSimd(Real3{0_r, 0_r, 0_r});
  Vec4r const X1 = ToSimd(Real3{1_r, 0_r, 0_r});
  Vec4r const frameAxis = ToSimd(Real3{0_r, 1_r, 0_r});

  // Test with various DOF configurations
  DynamicArray<DynamicArray<real>> const dofConfigs = {
      {0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r},
      {0.1_r, 0_r, 0_r, 0_r, 0_r, 0.1_r, 0_r, 0_r},
      {0_r, 0.1_r, 0_r, 0.2_r, 0_r, 0_r, 0.1_r, 0.1_r},
      {0.05_r, 0.05_r, 0.05_r, 0.1_r, 0.05_r, 0.05_r, 0.05_r, 0.2_r}};

  DynamicArray<Quaternion> const rotations = {
      kqI, quatx(ToRads(45_r)), quaty(ToRads(-30_r)), quatz(ToRads(60_r))};

  DynamicArray<TransformRT> const transforms = {
      TransformRT(kqI, kv0),
      TransformRT(quatx(ToRads(20_r)), kvx),
      TransformRT(Quaternion::FromAxisAngle(Normalize(kv123), ToRads(15_r)), kvy)};

  DynamicArray<real> const zeroDofs(8, 0_r);
  for (auto const& rodDofs : dofConfigs) {
    // Transport the reference frame axis to be consistent with the deformed tangent
    Vec4r const consistentAxis = test::ComputePerturbedFrameAxis(
        X0, X1, MakeConstSpan(zeroDofs), MakeConstSpan(rodDofs), frameAxis);
    for (auto const& rotRigid : rotations) {
      for (auto const& wflRod : transforms) {
        Quaternion const refFrame = quatx(ToRads(10_r));
        Quaternion const targetRot = quaty(ToRads(15_r));
        testFD(rotRigid, wflRod, X0, X1, consistentAxis, rodDofs, refFrame, targetRot);
      }
    }
  }
}

TEST(Constraints, RodElementRotationToRigidJacobianTarget) {
  auto testFD = [](Quaternion const& rotRigid,
                   TransformRT const& worldFromLocalRod,
                   Vec4r const& X0Rod,
                   Vec4r const& X1Rod,
                   Vec4r const& frameAxisRod,
                   DynamicArray<real> const& rodElementDofs,
                   Quaternion const& refFrame,
                   Quaternion targetRot) {
    real constexpr kDelta = 1.0e-2_r;
    real constexpr kTol = 1.0e-2_r;
    ColumnVector<real, 3> val;
    RowMatrix<real, 3, 3> JacTarget;

    EvalRodElementRotationToRigidConstraint(
        rotRigid,
        worldFromLocalRod,
        X0Rod,
        X1Rod,
        frameAxisRod,
        rodElementDofs,
        refFrame,
        targetRot,
        &val,
        nullptr,
        &JacTarget);

    auto const val0 = val.Duplicate();
    auto JacFD = JacTarget.Duplicate();

    // Test derivative w.r.t. target rotation
    for (int i = 0; i < 3; ++i) {
      Real3 delta{};
      delta[i] = kDelta;
      Quaternion const targetRotP = Quaternion::FromRotationVector(delta) * targetRot;
      EvalRodElementRotationToRigidConstraint(
          rotRigid,
          worldFromLocalRod,
          X0Rod,
          X1Rod,
          frameAxisRod,
          rodElementDofs,
          refFrame,
          targetRotP,
          &val,
          nullptr,
          nullptr);
      auto valp = val.Duplicate();

      delta[i] = -kDelta;
      Quaternion const targetRotM = Quaternion::FromRotationVector(delta) * targetRot;
      EvalRodElementRotationToRigidConstraint(
          rotRigid,
          worldFromLocalRod,
          X0Rod,
          X1Rod,
          frameAxisRod,
          rodElementDofs,
          refFrame,
          targetRotM,
          &val,
          nullptr,
          nullptr);
      auto const valm = val.Duplicate();

      JacFD.Col(i) = SelectClosest(val0, valp, valm, JacTarget.Col(i), kDelta);
    }

    EXPECT_LE((RowMatrix<real, 3, 3>(JacFD - JacTarget)).Norm(), kTol * JacFD.Norm());
  };

  // Setup test data for rod element
  Vec4r const X0 = ToSimd(Real3{0_r, 0_r, 0_r});
  Vec4r const X1 = ToSimd(Real3{1_r, 0_r, 0_r});
  Vec4r const frameAxis = ToSimd(Real3{0_r, 1_r, 0_r});

  // Test with various DOF configurations
  DynamicArray<DynamicArray<real>> const dofConfigs = {
      {0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r},
      {0.1_r, 0_r, 0_r, 0_r, 0_r, 0.1_r, 0_r, 0_r},
      {0_r, 0.1_r, 0_r, 0.2_r, 0_r, 0_r, 0.1_r, 0.1_r},
      {0.05_r, 0.05_r, 0.05_r, 0.1_r, 0.05_r, 0.05_r, 0.05_r, 0.2_r}};

  DynamicArray<Quaternion> const rotations = {
      kqI, quatx(ToRads(45_r)), quaty(ToRads(-30_r)), quatz(ToRads(60_r))};

  DynamicArray<TransformRT> const transforms = {
      TransformRT(kqI, kv0),
      TransformRT(quatx(ToRads(20_r)), kvx),
      TransformRT(Quaternion::FromAxisAngle(Normalize(kv123), ToRads(15_r)), kvy)};

  DynamicArray<real> const zeroDofs(8, 0_r);
  for (auto const& rodDofs : dofConfigs) {
    Vec4r const consistentAxis = test::ComputePerturbedFrameAxis(
        X0, X1, MakeConstSpan(zeroDofs), MakeConstSpan(rodDofs), frameAxis);
    for (auto const& rotRigid : rotations) {
      for (auto const& wflRod : transforms) {
        Quaternion const refFrame = quatx(ToRads(10_r));
        Quaternion const targetRot = quaty(ToRads(15_r));
        testFD(rotRigid, wflRod, X0, X1, consistentAxis, rodDofs, refFrame, targetRot);
      }
    }
  }
}
