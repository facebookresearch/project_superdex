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

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/half.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/simd.h>

#include <gtest/gtest.h>

#include <functional>
#include <type_traits>
#include <vector>

#if defined(_MSC_VER) && !defined(__AVX2__)
#error For better performance, please enable AVX2 support. Use MSVC flag = "/arch:AVX2", GCC/Clang flag = "-mavx2".
#endif

namespace mochi::simd_test {

// Short hand
static constexpr real kEps = kDefaultNearEqualEpsilon<real>;

// Return true if (a == b) or if a and b have the same bits (in case of nan)
template <class T>
static bool EqualValueOrBits(T a, T b) {
  return (a == b) || (0 == memcmp(&a, &b, sizeof(T)));
}

// Return true if NearEqual(a, b, eps) or if a and b have the same bits (in case of nan)
template <class T>
static bool NearEqualValueOrBits(T a, T b, T eps) {
  return NearEqual(a, b, eps) || (0 == memcmp(&a, &b, sizeof(T)));
}

template <class T, class VecType>
static void TestUnaryFunction(
    std::function<T(T)> scalarFn,
    std::function<VecType(VecType)> vectorFn,
    std::function<bool(T, T)> compareFn,
    T min = T(-VecType::kSize),
    T max = T(VecType::kSize)) {
  ASSERT_GE(max, min);
  T const kValues[] = {
      T(min + 0.05 * (max - min)),
      T(min + 0.95 * (max - min)),
      T(min + 0.4 * (max - min)),
      T(min + 0.6 * (max - min)),
      T(min + 0.3 * (max - min)),
      T(min + 0.7 * (max - min)),
      T(min + 0.45 * (max - min)),
      T(min + 0.55 * (max - min))};
  auto result = vectorFn(Load<VecType>(&kValues[0]));
  for (int i = 0; i < VecType::kSize; ++i) {
    T expected = scalarFn(kValues[i]);
    T actual = Get(result, i);
    EXPECT_TRUE(compareFn(expected, actual)) << "exepcted " << expected << ", but got " << actual;
  }
  // Also test +/- zero
  T const kZeros[] = {T(0), -T(0)};
  for (auto const& val : kZeros) {
    result = vectorFn(VecType{val});
    T expected = scalarFn(val);
    for (int i = 0; i < VecType::kSize; ++i) {
      T actual = Get(result, i);
      EXPECT_TRUE(compareFn(expected, actual));
    }
  }
}

// Test a unary Simd function by comparing it to a known scalar function, for a list of user-defined
// test values. Usef ro
template <class SimdT>
inline void TestSimdTrigFunction(
    std::function<double(double)> expectedScalarFn,
    std::function<SimdT(SimdT)> actualSimdFn,
    Span<double const> values) {
  constexpr double kEpsilonSingle =
      1.2e-7; // Our custom implementation of sin/cos could use a smaller tolerance, but not SVML.
  constexpr double kEpsilonDouble = 1.0e-15;
  constexpr double kEpsilon =
      std::is_same_v<double, typename SimdT::Scalar> ? kEpsilonDouble : kEpsilonSingle;
  [[maybe_unused]] double maxDiffValue = 0_r;
  [[maybe_unused]] double maxDiffAngle = 0_r;
  for (int i = 0; i + SimdT::kSize <= isize(values); ++i) {
    auto inputAngles = Load<Simd<double, SimdT::kSize>>(&values[i]);
    double expectedVec[SimdT::kSize];
    for (int j = 0; j < SimdT::kSize; ++j) {
      expectedVec[j] = expectedScalarFn(inputAngles[j]);
    }
    SimdT actualVec = actualSimdFn(StaticCast<SimdT>(inputAngles));
    for (int j = 0; j < SimdT::kSize; ++j) {
      double angle = inputAngles[j];
      double expected = expectedVec[j];
      double actual = actualVec[j];
      EXPECT_NEAR(expected, actual, kEpsilon) << "At value: " << angle;
      double diff = Abs(expected - actual);
      if (diff > maxDiffValue) {
        maxDiffValue = diff;
        maxDiffAngle = angle;
      }
    }
  }

  // Uncommnet to print the max error
  // printf("Max error = %.17g, at value %.17g\n", maxDiffValue, maxDiffAngle);
}

template <class T, class VecType>
static void TestBinaryFunction(
    std::function<T(T, T)> const& scalarFn,
    std::function<VecType(VecType, VecType)> vectorFn,
    std::function<bool(T, T)> compareFn) {
  constexpr T kValuesA[] = {T(1), T(-2.25), T(3.5), T(-4.75), T(5.25), T(-6.5), T(7.75), T(-8)};
  constexpr T kValuesB[] = {T(-2.25), T(3.5), T(4.75), T(-5.75), T(-6.5), T(7.25), T(8), T(-9)};
  auto va = Load<VecType>(&kValuesA[0]);
  auto vb = Load<VecType>(&kValuesB[0]);
  auto result = vectorFn(va, vb);
  for (int i = 0; i < VecType::kSize; ++i) {
    T expected = scalarFn(kValuesA[i], kValuesB[i]);
    T actual = Get(result, i);
    EXPECT_TRUE(compareFn(expected, actual));
  }
}

template <class T, class VecType>
static void TestTernaryFunction(
    std::function<T(T, T, T)> const& scalarFn,
    std::function<VecType(VecType, VecType, VecType)> vectorFn,
    std::function<bool(T, T)> compareFn) {
  constexpr T kValuesA[] = {T(1), T(-2.25), T(3.5), T(-4.75), T(5.25), T(-6.5), T(7.75), T(-8)};
  constexpr T kValuesB[] = {T(-2.25), T(3.5), T(4.75), T(-5.75), T(-6.5), T(7.25), T(8), T(-9)};
  constexpr T kValuesC[] = {T(8), T(7.75), T(6.5), T(5.25), T(-4.75), T(-3.5), T(-2.25), T(-1)};
  auto va = Load<VecType>(&kValuesA[0]);
  auto vb = Load<VecType>(&kValuesB[0]);
  auto vc = Load<VecType>(&kValuesC[0]);
  auto result = vectorFn(va, vb, vc);
  for (int i = 0; i < VecType::kSize; ++i) {
    T expected = scalarFn(kValuesA[i], kValuesB[i], kValuesC[i]);
    T actual = Get(result, i);
    EXPECT_TRUE(compareFn(expected, actual));
  }
}

#define MOCHI_SIMD_TEST_UNARY_FN_NEAR(VecType, VecFn, ScalarFn, eps)                      \
  TEST(VecType, VecFn) {                                                                  \
    TestUnaryFunction<VecType::Scalar, VecType>(                                          \
        [](auto a) { return ScalarFn(a); },                                               \
        [](auto a) { return VecFn(a); },                                                  \
        [](auto a, auto b) { return NearEqualValueOrBits(a, b, VecType::Scalar(eps)); }); \
  }

#define MOCHI_SIMD_TEST_LIMIT_RANGE_UNARY_FN_NEAR(VecType, VecFn, ScalarFn, min, max, eps) \
  TEST(VecType, VecFn) {                                                                   \
    TestUnaryFunction<VecType::Scalar, VecType>(                                           \
        [](auto a) { return ScalarFn(a); },                                                \
        [](auto a) { return VecFn(a); },                                                   \
        [](auto a, auto b) { return NearEqualValueOrBits(a, b, VecType::Scalar(eps)); },   \
        VecType::Scalar(min),                                                              \
        VecType::Scalar(max));                                                             \
  }

#define MOCHI_SIMD_TEST_UNARY_FN_EXACT(VecType, VecFn, ScalarFn) \
  TEST(VecType, VecFn) {                                         \
    TestUnaryFunction<VecType::Scalar, VecType>(                 \
        [](auto a) { return ScalarFn(a); },                      \
        [](auto a) { return VecFn(a); },                         \
        [](auto a, auto b) { return EqualValueOrBits(a, b); });  \
  }

// Test binary function. Expact approximate results.
#define MOCHI_SIMD_TEST_BINARY_FN_NEAR(VecType, VecFn, ScalarFn, eps)                     \
  TEST(VecType, VecFn) {                                                                  \
    TestBinaryFunction<VecType::Scalar, VecType>(                                         \
        [](auto a, auto b) { return ScalarFn(a, b); },                                    \
        [](auto a, auto b) { return VecFn(a, b); },                                       \
        [](auto a, auto b) { return NearEqualValueOrBits(a, b, VecType::Scalar(eps)); }); \
  }

// Test binary function. Expect exact results.
#define MOCHI_SIMD_TEST_BINARY_FN_EXACT(VecType, VecFn, ScalarFn) \
  TEST(VecType, VecFn) {                                          \
    TestBinaryFunction<VecType::Scalar, VecType>(                 \
        [](auto a, auto b) { return ScalarFn(a, b); },            \
        [](auto a, auto b) { return VecFn(a, b); },               \
        [](auto a, auto b) { return EqualValueOrBits(a, b); });   \
  }

// Test ternary function. Expact approximate results.
#define MOCHI_SIMD_TEST_TERNARY_FN_NEAR(VecType, VecFn, ScalarFn, eps)                    \
  TEST(VecType, VecFn) {                                                                  \
    TestTernaryFunction<VecType::Scalar, VecType>(                                        \
        [](auto a, auto b, auto c) { return ScalarFn(a, b, c); },                         \
        [](auto a, auto b, auto c) { return VecFn(a, b, c); },                            \
        [](auto a, auto b) { return NearEqualValueOrBits(a, b, VecType::Scalar(eps)); }); \
  }

// Test unary bitwise operator. Expect exact bitwise results.
#define MOCHI_SIMD_TEST_UNARY_BITWISE_OP(VecType, OpName, Op)           \
  TEST(VecType, OpName) {                                               \
    TestUnaryFunction<VecType::Scalar, VecType>(                        \
        [](auto a) {                                                    \
          uint64_t ia;                                                  \
          memcpy(&ia, &a, sizeof(a));                                   \
          uint64_t ix = Op ia;                                          \
          VecType::Scalar result;                                       \
          memcpy(&result, &ix, sizeof(result));                         \
          return result;                                                \
        },                                                              \
        [](auto a) { return Op a; },                                    \
        [](auto a, auto b) { return memcmp(&a, &b, sizeof(a)) == 0; }); \
  }

// Test binary bitwise operator. Expect exact bitwise results.
#define MOCHI_SIMD_TEST_BINARY_BITWISE_OP(VecType, OpName, Op)          \
  TEST(VecType, OpName) {                                               \
    TestBinaryFunction<VecType::Scalar, VecType>(                       \
        [](auto a, auto b) {                                            \
          uint64_t ia, ib;                                              \
          memcpy(&ia, &a, sizeof(a));                                   \
          memcpy(&ib, &b, sizeof(b));                                   \
          uint64_t ix = ia Op ib;                                       \
          VecType::Scalar result;                                       \
          memcpy(&result, &ix, sizeof(result));                         \
          return result;                                                \
        },                                                              \
        [](auto a, auto b) { return a Op b; },                          \
        [](auto a, auto b) { return memcmp(&a, &b, sizeof(a)) == 0; }); \
  }

// Test binary logical operator on SIMD masks. Verifies that for the four broadcast combinations
// of true (all-bits-1) and false (all-bits-0) operands, LogicalOp produces the same bits as the
// corresponding BitwiseOp.
#define MOCHI_SIMD_TEST_BINARY_LOGICAL_OP(VecType, OpName, LogicalOp, BitwiseOp) \
  TEST(VecType, OpName) {                                                        \
    auto const vFalse = SimdZero<VecType>();                                     \
    auto const vTrue = ~vFalse;                                                  \
    auto const checkOp = [](VecType a, VecType b) {                              \
      VecType const logical = a LogicalOp b;                                     \
      VecType const bitwise = a BitwiseOp b;                                     \
      EXPECT_TRUE(memcmp(&logical, &bitwise, sizeof(VecType)) == 0);             \
    };                                                                           \
    checkOp(vTrue, vTrue);                                                       \
    checkOp(vTrue, vFalse);                                                      \
    checkOp(vFalse, vTrue);                                                      \
    checkOp(vFalse, vFalse);                                                     \
  }

// Tests details::IsValidLogicalMask: returns true for all-bits-0 and all-bits-1 masks
// (per lane), false for any lane with a partial bit pattern.
#define MOCHI_SIMD_TEST_IS_VALID_LOGICAL_MASK(VecType)                                        \
  TEST(VecType, IsValidLogicalMask) {                                                         \
    auto const vFalse = SimdZero<VecType>();                                                  \
    auto const vTrue = ~vFalse;                                                               \
    EXPECT_TRUE(mochi::details::IsValidLogicalMask(vFalse));                                  \
    EXPECT_TRUE(mochi::details::IsValidLogicalMask(vTrue));                                   \
    /* Corrupt one lane at a time and verify the per-lane reduction catches it.            */ \
    /* Use memcpy (not reinterpret_cast<Scalar*>) to side-step strict-aliasing UB:         */ \
    for (int i = 0; i < VecType::kSize; ++i) {                                                \
      auto vMixed = vFalse;                                                                   \
      typename VecType::Scalar const laneValue{1};                                            \
      memcpy(                                                                                 \
          reinterpret_cast<char*>(&vMixed) + i * sizeof(laneValue),                           \
          &laneValue,                                                                         \
          sizeof(laneValue));                                                                 \
      EXPECT_FALSE(mochi::details::IsValidLogicalMask(vMixed));                               \
    }                                                                                         \
  }

// Test unary operator. Expect exact results.
#define MOCHI_SIMD_TEST_UNARY_OP_EXACT(VecType, OpName, Op)     \
  TEST(VecType, OpName) {                                       \
    TestUnaryFunction<VecType::Scalar, VecType>(                \
        [](auto a) { return Op a; },                            \
        [](auto a) { return Op a; },                            \
        [](auto a, auto b) { return EqualValueOrBits(a, b); }); \
  }

// Test binary operator. Expact approximate results.
#define MOCHI_SIMD_TEST_BINARY_OP_NEAR(VecType, OpName, Op, eps)                          \
  TEST(VecType, OpName) {                                                                 \
    TestBinaryFunction<VecType::Scalar, VecType>(                                         \
        [](auto a, auto b) { return a Op b; },                                            \
        [](auto a, auto b) { return a Op b; },                                            \
        [](auto a, auto b) { return NearEqualValueOrBits(a, b, VecType::Scalar(eps)); }); \
  }

// Test binary operator. Expect exact results.
#define MOCHI_SIMD_TEST_BINARY_OP_EXACT(VecType, OpName, Op)    \
  TEST(VecType, OpName) {                                       \
    TestBinaryFunction<VecType::Scalar, VecType>(               \
        [](auto a, auto b) { return a Op b; },                  \
        [](auto a, auto b) { return a Op b; },                  \
        [](auto a, auto b) { return EqualValueOrBits(a, b); }); \
  }

#define MOCHI_SIMD_TEST_SCALAR_CONVERSIONS(VecType)                                 \
  /* Conversion from bool is explicitly disallowed */                               \
  static_assert(!std::is_constructible_v<VecType, bool>);                           \
  static_assert(!std::is_assignable_v<VecType&, bool>);                             \
  /* Other conversions are allowed (consistent with C++ scalar conversion rules) */ \
  static_assert(std::is_constructible_v<VecType, Half>);                            \
  static_assert(std::is_constructible_v<VecType, int>);                             \
  static_assert(std::is_constructible_v<VecType, int64_t>);                         \
  static_assert(std::is_constructible_v<VecType, float>);                           \
  static_assert(std::is_constructible_v<VecType, double>);                          \
  static_assert(std::is_assignable_v<VecType&, Half>);                              \
  static_assert(std::is_assignable_v<VecType&, int>);                               \
  static_assert(std::is_assignable_v<VecType&, int64_t>);                           \
  static_assert(std::is_assignable_v<VecType&, float>);                             \
  static_assert(std::is_assignable_v<VecType&, double>);

#define MOCHI_SIMD_TEST_SCALAR_CONVERSIONS_TO_HALF(VecType)  \
  /* Conversion from bool is explicitly disallowed */        \
  static_assert(!std::is_constructible_v<VecType, bool>);    \
  static_assert(!std::is_assignable_v<VecType&, bool>);      \
  /* Conversion from Half is OK (broadcast) */               \
  static_assert(std::is_constructible_v<VecType, Half>);     \
  static_assert(std::is_assignable_v<VecType&, Half>);       \
  /* Other conversions require an explicit cast to Half */   \
  static_assert(!std::is_constructible_v<VecType, int>);     \
  static_assert(!std::is_constructible_v<VecType, int64_t>); \
  static_assert(!std::is_constructible_v<VecType, float>);   \
  static_assert(!std::is_constructible_v<VecType, double>);  \
  static_assert(!std::is_assignable_v<VecType&, int>);       \
  static_assert(!std::is_assignable_v<VecType&, int64_t>);   \
  static_assert(!std::is_assignable_v<VecType&, float>);     \
  static_assert(!std::is_assignable_v<VecType&, double>);

// Test Equal<N>/NotEqual<N> on a SIMD vector pair, plus the operator forms when N == V::kSize.
// NotEqual<N> for N > 1 routes through V::AnyTrue<N>, which is not implemented for Simd<int64_t,N>
// (Vec2l, Vec4l). For Vec4l (composite) the failure is deep inside AnyTrue's body, so a
// `requires` SFINAE check cannot detect it. Use an explicit scalar-type gate that matches the
// original int64x2/int64x4 special-case.
template <int N, class V>
static void ExpectEqual(bool expect, V a, V b) {
  EXPECT_EQ(expect, Equal<N>(a, b));
  if constexpr (N == 1 || !std::is_same_v<typename V::Scalar, int64_t>) {
    EXPECT_EQ(!expect, NotEqual<N>(a, b));
  }
  if constexpr (N == V::kSize) {
    EXPECT_EQ(expect, (a == b));
    EXPECT_EQ(!expect, (a != b));
  }
}

// Build a list of values useful for testing functions like sin and cos.
inline std::vector<double> GetTrigTestValues() {
  std::vector<double> vals;
  vals.reserve((32 + static_cast<int>(6 * kPI / 0.001)) * 2);

  // Start with angles on exact 45 degree boundaries
  for (int i = 0; i < 32; ++i) {
    vals.push_back((kPI * 0.25) * i);
  }

  // Add many more angles at fine subdivisions
  for (double r = 0_r; r < 4 * kPI; r += 0.001) {
    vals.push_back(r);
  }

  // Some large values
  constexpr double kLargeValue = 1.0e6;
  for (double r = 0_r; r < 2 * kPI; r += 0.001) {
    vals.push_back(r + kLargeValue);
  }

  // Duplicate all values and flip the sign
  int n = isize(vals);
  vals.reserve(2 * n);
  vals.insert(vals.end(), vals.begin(), vals.end());
  for (int i = n; i < isize(vals); ++i) {
    vals[i] = -vals[i];
  }

  return vals;
}

} // namespace mochi::simd_test
