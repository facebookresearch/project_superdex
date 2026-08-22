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

#pragma once

#include <mochi_core/mochi_config.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/simd.h>

#include <cmath>

namespace mochi {

/**
  Soft Plus (for smoothing discontinuities in signed distance)

  The function and the derivatives of the SoftPlus function, written in
  their numerically-stable form (algebraically equivalent to the textbook
  f(x) = ln(1 + exp(tx))/t).

  f(x) = (max(0, tx) + log1p(exp(-|tx|)))/t
  f'(x) = 1/(1 + exp(-tx))
  f''(x) = t*f'(x)*(1 - f'(x))
 */
template <class T, int kDerivative = 0>
[[nodiscard]] MOCHI_FORCE_INLINE T SoftPlus(T x, T t) {
  if constexpr (kDerivative == 0) {
    // Numerically stable SoftPlus identity:
    //   ln(1 + exp(tx)) = max(0, tx) + ln(1 + exp(-|tx|))
    T const tx = t * x;
    return (Max(tx, (T)0) + std::log1p(std::exp(-Abs(tx)))) / t;
  } else if constexpr (kDerivative == 1) {
    return (T)1 / ((T)1 + std::exp(-t * x));
  } else if constexpr (kDerivative == 2) {
    T const s = SoftPlus<T, 1>(x, t);
    return t * s * ((T)1 - s);
  } else {
    static_assert(
        (kDerivative >= 0) && (kDerivative < 3),
        "Derivatives higher than 2 not implemented for SoftPlus");
  }
}

/**
  A general clamping function with clamping parameter alpha.
  If alpha <= 0, the function is the identity (no clamping).
  If alpha > 0 but is finite, the function is a smooth approximation to the clamping function.
  In this case we are using the smooth clamping function:
    f(x)=(log(1+exp(alpha*x)) - log(1+exp(alpha*(x-1)))) / alpha
  If alpha is infinite, the function is the exact clamping function.
 */
template <class T>
MOCHI_FORCE_INLINE T SoftClamp(T value, T alpha, T* outDClampedValue = nullptr) {
  T zero = 0, one = 1;
  auto const maskNoClamp = alpha <= zero;
  auto const maskExactClamp = !IsFinite(alpha);
  T outClampedValue = Select(
      maskNoClamp,
      value,
      Select(
          maskExactClamp,
          Clamp(value, zero, one),
          SoftPlus<T, 0>(value, alpha) - SoftPlus<T, 0>(value - one, alpha)));
  if (outDClampedValue) {
    *outDClampedValue = Select(
        maskNoClamp,
        one,
        Select(
            maskExactClamp,
            Select(outClampedValue == value, one, zero),
            SoftPlus<T, 1>(value, alpha) - SoftPlus<T, 1>(value - one, alpha)));
  }
  return outClampedValue;
}

/**
  @brief: PolyReLU - A smoothed ReLU activation function.

  Evaluates a polynomial (in y = (x - shift) / t):

    f(y) = 3/16*t + (1/2 + 3/8*y - 1/16*y^3) * (x - shift)

  such that

    f(shift-t) = 0; f'(shift-t) = 0; f(shift+t) = t; f'(shift+t) = 1; f''(shift+t) = 0

  Effectively it's a C2-continuous approximation to ReLU centered at `shift`,
  transitioning smoothly from 0 to linear over the interval (shift-t, shift+t).

  @param[in] x the coordinate where to evaluate the PolyReLU
  @param[in] t the half-width of the transition zone
  @param[in] shift the center of the transition zone
  @param[out] f function value
  @param[out] df first derivative
  @param[out] ddf second derivative
 */
template <class T>
MOCHI_FORCE_INLINE void PolyReLU(T x, T t, T shift, T& f, T& df, T& ddf) {
  x -= shift;
  T const x_t = x / t;
  T const x2_t2 = x_t * x_t;
  T const x3_t3 = x2_t2 * x_t;
  T const zero = 0, one = 1;
  auto const mask = (t <= zero) | (Abs(x) > t);
  f = Max(
      zero,
      Select(
          mask, x, T(3.0 / 16.0) * t + (T(0.5) + T(3.0 / 8.0) * x_t - T(1.0 / 16.0) * x3_t3) * x));
  df = Clamp(
      Select(mask, Select(x < zero, zero, one), T(0.5) + T(0.75) * x_t - T(0.25) * x3_t3),
      zero,
      one);
  ddf = Max(zero, Select(mask, zero, T(0.75) * (one - x2_t2) / t));
}

/**************************************************************************************************
 IPC unit step function C1 interpolant of forces, as defined in Incremental Potential Contact,
 LiEtAl20. Given some input x in the range [0, t], the derivative of this function yields a C1
 interpolant of the unit step:

 if (x >= t)
    df = 1
 if (x in [0, t))
    df = -x^2/t^2 + 2x/t

 This function outputs f as it is necessary for the computation of the friction potential, and not
 just df.

 f: function
 df: derivative, interpolant of the unit step
 ddf: second derivative, derivative of the interpolant
 df_x: interpolant divided by x, for robustness under small values of x
*/
template <class T>
MOCHI_FORCE_INLINE void IPCstepC1(T const& x, T const& t, T& f, T& df, T& ddf, T& df_x) {
  static_assert(std::is_floating_point_v<T> || IsSimd<T>, "Unsupported type for IPCstepC1");
  if constexpr (IsSimd<T>) {
    static_assert(std::is_floating_point_v<typename T::Scalar>, "Unsupported type for IPCstepC1");
  }
  MOCHI_ASSERT_VERBOSE(AllTrue(x >= T(0)), "IPCstepC1 only supported for non-negative values.");
  MOCHI_ASSERT_VERBOSE(AllTrue(t >= T(0)), "Falloff distance must be non-negative.");
  T zero = T{0}, one = T{1}, two = T{2}, oneThird = (T{1} / T{3});
  T x_t = x / t;
  auto const mask = (x >= t);
  auto const isZero = [&]() {
    if constexpr (IsSimd<T>) {
      return VEqual(x, zero);
    } else {
      return (x == zero);
    }
  }();
  f = Select(mask, x - oneThird * t, x * x_t * (one - oneThird * x_t));
  ddf = Select(mask, zero, two * (one - x_t) / t);
  df_x = Select(mask, Select(isZero, zero, one / x), (two - x_t) / t);
  df = Select(mask, one, x * df_x);
}

/**************************************************************************************************
 C-infinity regularized friction activation function. Given some input x >= 0 and
 regularization parameter eps >= 0, this function computes:

 f(x)    = sqrt(x^2 + eps^2/4) - eps/2            // energy smoother, shifted so f(0) = 0
 df(x)   = x / sqrt(x^2 + eps^2/4)                // first derivative
 ddf(x)  = (eps^2/4) / (x^2 + eps^2/4)^{3/2}      // second derivative
 df_x(x) = 1 / sqrt(x^2 + eps^2/4)                // df/x (robust, no division by x)

 This is a C-infinity smooth approximation to f(x) = |x|, which converges as eps -> 0.
 Unlike IPCstepC1, the regularization has no compact support: the function smoothly
 transitions everywhere, with the steepest transition near x = 0.

 The eps parameter is clamped internally to a small positive value to avoid division by zero
 when eps = 0. The value of f is shifted by -eps/2 so that f(0) = 0, matching the
 convention in other similar activation functions.
*/
template <class T>
MOCHI_FORCE_INLINE void CinfRegularized(T const& x, T const& eps, T& f, T& df, T& ddf, T& df_x) {
  static_assert(std::is_floating_point_v<T> || IsSimd<T>, "Unsupported type for CinfRegularized");
  if constexpr (IsSimd<T>) {
    static_assert(
        std::is_floating_point_v<typename T::Scalar>, "Unsupported type for CinfRegularized");
  }
  MOCHI_ASSERT_VERBOSE(
      AllTrue(x >= T(0)), "CinfRegularized only supported for non-negative values.");
  // Clamp eps to avoid division by zero when the caller passes eps = 0.
  T const epsClamped = Max(eps, T{std::numeric_limits<real>::epsilon()});
  // The factor of 1/4 (i.e., dividing eps by 2) ensures that ddf(0) = 2/eps, matching the
  // second derivative of IPCstepC1 at x = 0 (with t = eps) for consistent regularization.
  T const eps2 = epsClamped * epsClamped / 4_r;
  T const s2 = x * x + eps2;
  T const s = Sqrt(s2);
  T const invS = T{1} / s;

  f = s - epsClamped / 2_r; // energy smoother, shifted so f(0) = 0
  df = x * invS; // first derivative
  ddf = eps2 * (invS * invS * invS); // eps^2 / s^3
  df_x = invS; // 1/s (robust, no division by x)
}

// Interpreting the argument x as a velocity, df is a Gaussian Stribeck friction model, but
// normalized for a peak static friction of 1 and shifted to the right by a regularization threshold
// t, to make room for smoothly ramping up from zero. For dfInfty = 1, this is equivalent to the IPC
// C^1 step model (which is also approached in the limit of xStribeck --> infty).
template <class T>
MOCHI_FORCE_INLINE void StribeckActivation(
    T const& x,
    T const& t,
    T const& dfInfty, // Value of df as x --> infty
    T const& xStribeck, // Governs rate at which df --> dfInf as x increases
    T& f,
    T& df,
    T& ddf,
    T& df_x) {
  // Currently only supporting scalar types, since this is the only case currently needed. Extending
  // this to SIMD would require implementing an Erf() function with SIMD support.
  static_assert(std::is_floating_point_v<T>, "Unsupported type for StribeckActivation");

  MOCHI_ASSERT_VERBOSE(x >= 0_r, "Stribeck activation requires non-negative argument.");
  MOCHI_ASSERT_VERBOSE(t >= 0_r, "Stribeck activation requires non-negative threshold.");
  MOCHI_ASSERT_VERBOSE(
      dfInfty >= 0_r && dfInfty <= 1_r,
      "Asymptotic derivative of Stribeck activation must be in [0,1].");
  MOCHI_ASSERT_VERBOSE(xStribeck >= 0_r, "Stribeck decay parameter must be non-negative.");
  // Below the threshold t, use the IPC step function.
  if (x < t) {
    // Equivalent to sub-threshold branch of IPC C^1 step.
    T const inv_t = 1_r / t;
    T const x_t = x * inv_t;
    f = x * x_t * (1_r - (1_r / 3_r) * x_t);
    ddf = 2_r * (1_r - x_t) * inv_t;
    df_x = (2_r - x_t) * inv_t;
    df = x * df_x;
  } else {
    T const xShifted = x - t;
    T const ipcStepIntegral = t * (2_r / 3_r); // IPC smoother evaluated at x = t.
    if (xStribeck > 0) {
      T const dfDrop = 1_r - dfInfty;
      T const xRatio = xShifted / xStribeck;
      T constexpr erfScale = 0.5_r * kSqrtPi;
      f = ipcStepIntegral + dfInfty * xShifted + dfDrop * xStribeck * erfScale * std::erf(xRatio);
      T const gaussianTerm = Exp(-Sqr(xRatio));
      df = dfInfty + dfDrop * gaussianTerm;
      ddf = -2_r * dfDrop * xRatio * gaussianTerm / xStribeck;
    } else {
      // Squeezing the width of the Gaussian to zero with xStribeck = 0 leads to a discontinuity in
      // df when dfInfty < 1, since the value should be 1 just below t.
      f = ipcStepIntegral + dfInfty * xShifted;
      df = dfInfty;
      ddf = 0_r;
    }
    if (x == 0_r)
      MOCHI_UNLIKELY {
        // Technically reachable if we allow t = 0 (cf. nested Select in IPCstepC1)
        df_x = 0_r;
      }
    else {
      df_x = df / x;
    }
  }
}

} // namespace mochi
