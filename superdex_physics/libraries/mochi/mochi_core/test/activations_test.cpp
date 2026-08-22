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
#include <mochi_core/utils/activations.h>

#include <mochi_core/test/mochi_test_helpers.h>

using namespace mochi;

using Scalar = real;

template <typename T>
static void TestPolyReLU() {
  T const kTol = 1e-2;
  T const kEps = MOCHI_USE_DOUBLE_PRECISION ? 1e-5 : 1e-4;
  for (T s = 0; AllTrue(s <= Scalar(1)); s += Scalar(0.1)) {
    for (T t = 1e-3; AllTrue(t < Scalar(1)); t += Scalar(1e-2)) {
      T const epst = kEps * t;
      T f{}, df{}, ddf{};
      PolyReLU<T>(-t + s * t, t, s * t, f, df, ddf);
      EXPECT_NEAR_TOL(T(0), f, kTol);
      EXPECT_NEAR_TOL(T(0), df, kTol);
      EXPECT_NEAR_TOL(T(0), ddf, kTol);
      PolyReLU<T>(t + s * t, t, s * t, f, df, ddf);
      EXPECT_NEAR_TOL(t, f, kTol);
      EXPECT_NEAR_TOL(T(1), df, kTol);
      EXPECT_NEAR_TOL(T(0), ddf, kTol);

      for (T x = -t / 2; AllTrue(x < t / Scalar(2)); x += (Scalar(2) * t / Scalar(20))) {
        PolyReLU<T>(x, t, s * t, f, df, ddf);
        T f_eps{}, df_eps{}, ddf_eps{};
        PolyReLU<T>(x + epst, t, s * t, f_eps, df_eps, ddf_eps);
        EXPECT_NEAR_TOL((f_eps - f) / epst, df, kTol * (T(1) + Abs(df)));
        EXPECT_NEAR_TOL((df_eps - df) / epst, ddf, kTol * (T(1) + Abs(ddf)));
      }
    }
  }
}

template <typename T>
static void TestIPCStepC1() {
  T const kTol = 1e-2;
  T const kEps = 1e-4;
  for (T t = 1e-3; AllTrue(t < Scalar(1)); t += Scalar(1e-2)) {
    T const epst = kEps * t;
    for (T x = 0; AllTrue(x < Scalar(2) * t); x += Scalar(2) * t / Scalar(20)) {
      // Analytic
      T valAnal = 0;
      T dvalAnal = 0;
      T dummy = 0;
      T dummy2 = 0;
      IPCstepC1<T>(x, t, dummy, valAnal, dvalAnal, dummy2);

      // Finite differences
      if (AllTrue(x - epst >= T(0))) {
        // Skip if x - epst < 0. IPCstepC1 is only defined for non-negative inputs.
        T valp = 0;
        T valm = 0;
        T intvalp = 0;
        T intvalm = 0;
        IPCstepC1<T>(x + epst, t, intvalp, valp, dummy, dummy2);
        IPCstepC1<T>(x - epst, t, intvalm, valm, dummy, dummy2);
        T valTest = (intvalp - intvalm) / (Scalar(2) * epst);
        EXPECT_NEAR_TOL(valTest, valAnal, kTol * (T(1) + Abs(valAnal)));

        if (AllTrue(Abs(x - t) >= epst)) {
          // Skip if |x - t| < epst.
          T dvalTest = (valp - valm) / (Scalar(2) * epst);
          EXPECT_NEAR_TOL(dvalTest, dvalAnal, kTol * (T(1) + Abs(dvalAnal)));
        }
      }
    }
  }
}

template <typename T>
static void TestCinfRegularized() {
  // Finite-difference parameters following the StribeckActivation pattern.
  T const kFdEps = MOCHI_USE_DOUBLE_PRECISION ? 1e-6 : 1e-4;
  T const kFdTol = T(1e3) * kFdEps;
  T const kTol = 1e-2;

  for (T eps = 1e-2; AllTrue(eps < Scalar(1)); eps += Scalar(1e-1)) {
    T const delta = kFdEps * eps;

    // Analytical values at x = 0: f = 0 (shifted), df = 0, ddf = 2/eps, df_x = 2/eps
    {
      T f{}, df{}, ddf{}, df_x{};
      CinfRegularized<T>(T(0), eps, f, df, ddf, df_x);
      EXPECT_NEAR_TOL(T(0), f, kTol * eps);
      EXPECT_NEAR_TOL(T(0), df, kTol);
      EXPECT_NEAR_TOL(T(2) / eps, ddf, kTol * (T(1) + T(2) / eps));
      EXPECT_NEAR_TOL(T(2) / eps, df_x, kTol * (T(1) + T(2) / eps));
    }

    // Derivative consistency via finite differences in the regularization region.
    // Beyond ~2*eps, ddf becomes small and the dfP-df subtraction loses float precision.
    for (T x = eps / Scalar(10); AllTrue(x < Scalar(2) * eps); x += eps / Scalar(10)) {
      T f{}, df{}, ddf{}, df_x{};
      CinfRegularized<T>(x, eps, f, df, ddf, df_x);

      T fP{}, dfP{}, ddfP{}, df_xP{};
      CinfRegularized<T>(x + delta, eps, fP, dfP, ddfP, df_xP);

      EXPECT_NEAR_TOL((fP - f) / delta, df, kFdTol * (T(1) + Abs(df)));
      EXPECT_NEAR_TOL((dfP - df) / delta, ddf, kFdTol * (T(1) + Abs(ddf)));
      EXPECT_NEAR_TOL(df / x, df_x, kTol * (T(1) + Abs(df_x)));
    }

    // Large x: f approaches x - eps/2, df approaches 1, ddf approaches 0, df_x approaches 1/x
    {
      T const x = Scalar(100) * eps;
      T f{}, df{}, ddf{}, df_x{};
      CinfRegularized<T>(x, eps, f, df, ddf, df_x);
      EXPECT_NEAR_TOL(x - eps / T(2), f, kTol * x);
      EXPECT_NEAR_TOL(T(1), df, kTol);
      EXPECT_NEAR_TOL(T(0), ddf, kTol * (T(1) + Abs(ddf)));
      EXPECT_NEAR_TOL(T(1) / x, df_x, kTol * (T(1) + T(1) / x));
    }
  }
}

TEST(Activations, SoftPlus) {
  // Derivative consistency via finite differences.
  for (real t = 1e-2_r; t < 1_r; t += 1e-2_r) {
    for (real x = -1_r; x < 1_r; x += 1e-2_r) {
      constexpr real kTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-5_r : 1e-2_r;
      constexpr real kEps = MOCHI_USE_DOUBLE_PRECISION ? 1e-6_r : 1e-3_r;
      real const f = SoftPlus(x, t);
      real const df = SoftPlus<real, 1>(x, t);
      real const ddf = SoftPlus<real, 2>(x, t);
      real const f_eps = SoftPlus(x + kEps, t);
      real const df_eps = SoftPlus<real, 1>(x + kEps, t);
      EXPECT_NEAR((f_eps - f) / kEps, df, kTol * (1_r + Abs(df)));
      EXPECT_NEAR((df_eps - df) / kEps, ddf, kTol * (1_r + Abs(ddf)));
    }
  }

  // Edge cases: large |t*x| must not produce inf/NaN.
  real const tValues[] = {0.1_r, 1_r, 10_r};
  for (real const t : tValues) {
    constexpr real kTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-10_r : 1e-5_r;
    real const xLarge = 1000_r / t; // t*x = 1000, well above exp(t*x) overflow threshold.

    // Large positive t*x: f(x) → x, f'(x) → 1, f''(x) → 0.
    EXPECT_NEAR(xLarge, SoftPlus(xLarge, t), kTol * xLarge);
    EXPECT_NEAR(1_r, (SoftPlus<real, 1>(xLarge, t)), kTol);
    EXPECT_NEAR(0_r, (SoftPlus<real, 2>(xLarge, t)), kTol);

    // Large negative t*x: f(x) → 0, f'(x) → 0, f''(x) → 0.
    EXPECT_NEAR(0_r, SoftPlus(-xLarge, t), kTol);
    EXPECT_NEAR(0_r, (SoftPlus<real, 1>(-xLarge, t)), kTol);
    EXPECT_NEAR(0_r, (SoftPlus<real, 2>(-xLarge, t)), kTol);
  }
}

TEST(Activations, SoftClamp) {
  constexpr double kTol = 1.e-6_r;
  constexpr double kEps = 1.e-6_r;
  auto testDerivative = [&](double alpha) {
    for (double x = -1; x < 1; x += 1.e-2) {
      double Dval{};
      auto val = SoftClamp<double>(x, alpha, &Dval);
      auto val_eps = SoftClamp<double>(x + kEps, alpha, nullptr);
      EXPECT_NEAR((val_eps - val) / kEps, Dval, kTol);
    }
  };
  // no clamping
  testDerivative(0_r);
  // finite clamping
  for (double alpha = 1.e-2; alpha < 1; alpha += 1.e-2) {
    testDerivative(alpha);
  }
  // exact clamping
  testDerivative(std::numeric_limits<double>::infinity());
}

TEST(Activations, PolyReLU) {
  TestPolyReLU<Scalar>();
  TestPolyReLU<Simd<Scalar>>();
  TestPolyReLU<Simd<Scalar, 2 * Simd<Scalar>::kSize>>();
  TestPolyReLU<Simd<Scalar, 4 * Simd<Scalar>::kSize>>();
}

TEST(Activations, IPCStepC1) {
  TestIPCStepC1<Scalar>();
  TestIPCStepC1<Simd<Scalar>>();
  TestIPCStepC1<Simd<Scalar, 2 * Simd<Scalar>::kSize>>();
  TestIPCStepC1<Simd<Scalar, 4 * Simd<Scalar>::kSize>>();
}

TEST(Activations, CinfRegularized) {
  TestCinfRegularized<Scalar>();
  TestCinfRegularized<Simd<Scalar>>();
  TestCinfRegularized<Simd<Scalar, 2 * Simd<Scalar>::kSize>>();
  TestCinfRegularized<Simd<Scalar, 4 * Simd<Scalar>::kSize>>();
}

// Tests for StribeckActivation
TEST(Activations, StribeckActivation_DerivativeConsistency_AboveThreshold) {
  // Test that derivatives are consistent in the x >= t branch using finite differences
  real constexpr kEps = MOCHI_USE_DOUBLE_PRECISION ? 1e-8_r : 1e-4_r;
  real constexpr kTol = 1e3_r * kEps;

  for (real t = 0.01_r; t < 0.5_r; t += 0.1_r) {
    // Scale finite difference epsilon by t to remain appropriate at various length scales.
    real const eps = kEps * t;
    for (real dfInfty = 0.0_r; dfInfty <= 1.0_r; dfInfty += 0.25_r) {
      // Avoid xStribeck << t, to ensure that t-based finite difference size remains appropriate
      for (real xStribeck = 0.7_r * t; xStribeck < 2_r; xStribeck += 0.3_r) {
        // Test points in the x >= t region
        for (real x = t; x < t + 1.0_r; x += 0.1_r) {
          real f{}, df{}, ddf{}, df_x{};
          StribeckActivation<real>(x, t, dfInfty, xStribeck, f, df, ddf, df_x);

          real fEps{}, dfEps{}, ddfEps{}, df_xEps{};
          StribeckActivation<real>(x + eps, t, dfInfty, xStribeck, fEps, dfEps, ddfEps, df_xEps);

          // Check df = d(f)/dx
          real dfNumeric = (fEps - f) / eps;
          EXPECT_NEAR_TOL(dfNumeric, df, kTol * (1_r + Abs(df)));

          // Check ddf = d(df)/dx
          real ddfNumeric = (dfEps - df) / eps;
          EXPECT_NEAR_TOL(ddfNumeric, ddf, kTol * (1_r + Abs(ddf)));

          // Check df_x = df / x
          if (x > eps) {
            EXPECT_NEAR_TOL(df / x, df_x, kTol * (1_r + Abs(df_x)));
          }
        }
      }
    }
  }
}

TEST(Activations, StribeckActivation_EquivalentToIPCStepC1_BelowThreshold) {
  // Test that for x < t, StribeckActivation outputs match IPCstepC1
  // Separate unit testing for IPCStepC1 then ensures derivative consistency below threshold.
  real constexpr kTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-10_r : 2e-5_r;

  for (real t = 0.01_r; t < 0.5_r; t += 0.1_r) {
    for (real dfInfty = 0.0_r; dfInfty <= 1.0_r; dfInfty += 0.25_r) {
      // Results for x < t *should* be independent of xStribeck; iterating values to be safe.
      for (real xStribeck = 0.01_r; xStribeck < 0.5_r; xStribeck += 0.1_r) {
        // Test points in the x < t region (including x = 0)
        for (real x = 0_r; x < t; x += t / 10_r) {
          real fStribeck{}, dfStribeck{}, ddfStribeck{}, df_xStribeck{};
          StribeckActivation<real>(
              x, t, dfInfty, xStribeck, fStribeck, dfStribeck, ddfStribeck, df_xStribeck);

          real fIpc{}, dfIpc{}, ddfIpc{}, df_xIpc{};
          IPCstepC1<real>(x, t, fIpc, dfIpc, ddfIpc, df_xIpc);
          EXPECT_NEAR_TOL(fIpc, fStribeck, kTol);
          EXPECT_NEAR_TOL(dfIpc, dfStribeck, kTol);
          EXPECT_NEAR_TOL(ddfIpc, ddfStribeck, kTol);
          EXPECT_NEAR_TOL(df_xIpc, df_xStribeck, kTol);
        }
      }
    }
  }
}

TEST(Activations, StribeckActivation_ContinuityAtThreshold) {
  // Test that f, df, and ddf are all continuous at x = t
  real constexpr kEps = MOCHI_USE_DOUBLE_PRECISION ? 1e-8_r : 1e-5_r;
  real constexpr kTol = 1e3_r * kEps;

  for (real t = 0.01_r; t < 0.5_r; t += 0.1_r) {
    real const eps = kEps * t;
    for (real dfInfty = 0.0_r; dfInfty <= 1.0_r; dfInfty += 0.25_r) {
      for (real xStribeck = 0.7_r * t; xStribeck < 2.0_r; xStribeck += 0.3_r) {
        // Evaluate just below t (x < t branch)
        real fBelow{}, dfBelow{}, ddfBelow{}, df_xBelow{};
        StribeckActivation<real>(
            t - eps, t, dfInfty, xStribeck, fBelow, dfBelow, ddfBelow, df_xBelow);

        // Evaluate just above t (x >= t branch)
        real fAbove{}, dfAbove{}, ddfAbove{}, df_xAbove{};
        StribeckActivation<real>(
            t + eps, t, dfInfty, xStribeck, fAbove, dfAbove, ddfAbove, df_xAbove);

        // f should be continuous
        EXPECT_NEAR_TOL(fBelow, fAbove, kTol);

        // df should be continuous and equal to 1
        EXPECT_NEAR_TOL(dfBelow, dfAbove, kTol);
        EXPECT_NEAR_TOL(dfBelow, 1_r, kTol);
        EXPECT_NEAR_TOL(dfAbove, 1_r, kTol);

        // ddf should be continuous
        EXPECT_NEAR_TOL(ddfBelow, ddfAbove, kTol);
      }
    }
  }
}
