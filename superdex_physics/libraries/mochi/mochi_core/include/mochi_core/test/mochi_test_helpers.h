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

#include <mochi_core/geometry/tetrahedral_mesh.h>
#include <mochi_core/mochi_config.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/log.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/span.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace mochi::test {

/**
  Expect that no error has been set on a mochi::Error object.
  Prints the error details if any.
*/
#define EXPECT_OK(error) EXPECT_TRUE((error).IsOK()) << "UNEXPECTED ERROR:\n%s" << error.ToString();

/**
  Expect that an error HAS been set on a mochi::Error object.
*/
#define EXPECT_NOT_OK(error) EXPECT_FALSE((error).IsOK()) << "EXPECTED AN ERROR";

/**
  If you have a test that should only be enabled in certain build configurations, then you can use:
    TEST_IF(condition, testFixtureName, testCaseName)

  Any disabled test names will be prefixed with "DISABLED_". When you run the test executable, it
  will warn your that some tests were disabled. Use argument "--gtest_also_run_disabled_test" to run
  them anyway.

  Example: To enable a test ONLY if MOCHI_USE_FEATURE_XXX is 1, you can write:

    TEST_IF(MOCHI_USE_FEATURE_XXX, MyTestCategory, MyTestName) {
      // Your test code here
    }

  Similarly, TEST_IF_F is a conditional TEST_F macro and TEST_IF_P is a conditional TEST_P macro.
  Unfortunately, the condition must be either 0 or 1. It cannot be a complex expression.
*/
#define TEST_IF(condition, fixture, name) \
  MOCHI_PP_CAT(MOCHI_TEST_IF_IMPL_, condition)(TEST, fixture, name)
#define TEST_IF_F(condition, fixture, name) \
  MOCHI_PP_CAT(MOCHI_TEST_IF_IMPL_, condition)(TEST_F, fixture, name)
#define TEST_IF_P(condition, fixture, name) \
  MOCHI_PP_CAT(MOCHI_TEST_IF_IMPL_, condition)(TEST_P, fixture, name)
#define MOCHI_TEST_IF_IMPL_0(macro, fixture, name) macro(fixture, DISABLED_##name)
#define MOCHI_TEST_IF_IMPL_1(macro, fixture, name) macro(fixture, name)

/**
  You can pass ExpectOK{} in place of an Error& function parameter. The test will fail if the
  function sets an error.
*/
class ExpectOK final {
 public:
  ~ExpectOK() {
    EXPECT_OK(_error);
  }
  operator Error&() {
    return _error;
  }
  Error _error;
};

/**
  You can pass ExpectNotOK{} in place of an Error& function parameter. The test will fail if the
  function sets an error.
*/
class ExpectNotOK final {
 public:
  ~ExpectNotOK() {
    EXPECT_NOT_OK(_error);
  }
  operator Error&() {
    return _error;
  }
  Error _error;
};

/**
  Expect mochi::NearEqual(a, b) is true. Works for floats, doubles, and NdArrays.
*/
#define EXPECT_NEAR_EQ(a, b) EXPECT_TRUE(::mochi::NearEqual((a), (b)))

/**
  Expect that the spans are the same size and have equal elements
*/
#define EXPECT_SPAN_EQ(a, b) EXPECT_TRUE(::mochi::test::EqualSpan((a), (b)))

/**
  Expect mochi::NearEqual(a, b, tol) is true. Works for floats, doubles, and NdArrays.
*/
#define EXPECT_NEAR_TOL(a, b, tol) EXPECT_TRUE(::mochi::NearEqual((a), (b), (tol)))

/**
  Expect that the differences between a and b are closer than the given relative threshold.
  Standard EXPECT_NEAR only tests for absolute epsilons, which may be not be robust for large
  values. References: http://realtimecollisiondetection.net/blog/?p=89
  https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
 */
// TODO: Move from std to our own functions (MathUtils).
#define EXPECT_NEAR_RTOL(a, b, tol) \
  EXPECT_NEAR(a, b, std::max(tol, (tol) * std::max(std::abs(a), std::abs(b))))

using TetMeshParams =
    std::pair<std::vector<Real3>, std::vector<Int4>>; // coordinates & connectivity
using TriMeshParams =
    std::pair<std::vector<Real3>, std::vector<Int3>>; // coordinates & connectivity

/**
Create a regular grid with one corner at (0,0,0) and specified scale and dimensions.
Node coordinates are created following the natural order of dimensions X, Y, Z:
       3 -------- 4 ------- 5
      / |       / |       / |
     /  |      /  |      /  |
    9 ------- 10 ------ 11  |
    |   0 ----|-- 1 ----|-- 2
    |  /      |  /      |  /
    | /       | /       | /
    6 ------- 7 ------- 8
Using default parameters, the result should be equal to CreateMinimalTetMeshUnitCube.
*/
TetMeshParams CreateMinimalTetMeshUnitGrid(
    Real3 scale = Real3{1_r, 1_r, 1_r},
    Int3 dims = Int3{1, 1, 1});

/**
  Create a solid unit cube with one corner at (0,0,0)

        2 ------- 3    8 coordinates (all on surface)
      / |       / |    5 tets (4 have surface faces)
     /  |      /  |
    6 ------- 7   |
    |   0 ----|-- 1
    |  /      |  /
    | /       | /
    4 ------- 5
*/
TetMeshParams CreateMinimalTetMeshUnitCube(Real3 scale = Real3{1_r, 1_r, 1_r});
TriMeshParams CreateMinimalTriMeshUnitCube(Real3 scale = Real3{1_r, 1_r, 1_r});

/* A unit tet with one corner at (0,0,0)

     3 _ _2
     | \ / \
     |  / \ \
     |/     \\
     0 -----1
*/
TetMeshParams CreateMinimalTetMeshSingleTet(Real3 scale = Real3{1_r, 1_r, 1_r});

/* A unit tet with one corner at (0,0,0) sharing a face with another tetrahedra of the cube
 */
TetMeshParams CreateMinimalTetMeshTwoShareFace(Real3 scale = Real3{1_r, 1_r, 1_r});

/* A unit tet with one corner at (0,0,0) sharing an edge with another tetrahedra of the cube
 */
TetMeshParams CreateMinimalTetMeshTwoShareEdge(Real3 scale = Real3{1_r, 1_r, 1_r});

/* A unit tet with one corner at (0,0,0) sharing a node with another tetrahedra of the cube
 */
TetMeshParams CreateMinimalTetMeshTwoShareNode(Real3 scale = Real3{1_r, 1_r, 1_r});

/**
  Create a unit triangle in the x-y plane with one corner at (0,0,0)

    2
    |\
    | \
    |  \
    0 - 1
 */
TriMeshParams CreateMinimalTriMeshSingleTri(Real2 scale = Real2{1_r, 1_r});

/**
  Take the parameters of a tetrahedral mesh and serialize them (in memory) into the file format
  supported by mochi::Context::ShapeCreateTetMeshFromJson.
*/
std::string SerializeTetMesh(TetMeshParams const& mesh);

/**
  Return true if the two Spans are equal length and have equal elements.
  Also works for std::vector or anything else with 'size' and 'data' members.
  For convenience, this template accepts any type that supports mochi::MakeSpan.
*/
template <typename SpanA, typename SpanB>
[[nodiscard]] bool EqualSpan(SpanA const& a_, SpanB const& b_) {
  auto a = MakeSpan(a_);
  auto b = MakeSpan(b_);
  if (a.size() != b.size()) {
    return false;
  } else if (a.data() != b.data()) {
    for (int i = 0; i < isize(a); ++i) {
      if (a[i] != b[i]) {
        return false;
      }
    }
  }
  return true;
}

template <
    typename T,
    typename U,
    typename E,
    MOCHI_CONCEPT(
        std::is_floating_point_v<T>&& std::is_floating_point_v<U>&& std::is_floating_point_v<E>)>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr auto NearEqual(T a, U b, E epsilon) {
  // Use a common type for the computation to avoid mixed-type arithmetic issues
  using CommonType = std::common_type_t<T, U, E>;
  auto const commonA = static_cast<CommonType>(a);
  auto const commonB = static_cast<CommonType>(b);
  auto const commonEps = static_cast<CommonType>(epsilon);
  return ::mochi::NearEqual(commonA, commonB, commonEps);
}

template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr auto NearEqual(T const& a, T const& b, T epsilon) {
  return ::mochi::NearEqual(a, b, epsilon);
}

/**
  Return true if the two Spans are equal length and have elements that are nearly equal (within
  epsilon). For convenience, this template accepts any type that supports mochi::MakeSpan.
*/
template <typename SpanA, typename SpanB, typename E>
[[nodiscard]] bool NearEqualSpan(SpanA const& a_, SpanB const& b_, E epsilon) {
  auto a = MakeSpan(a_);
  auto b = MakeSpan(b_);
  if (a.size() != b.size()) {
    return false;
  }
  //
  using ScalarA = std::decay_t<typename decltype(a)::value_type>;
  using ScalarB = std::decay_t<typename decltype(b)::value_type>;
  if constexpr (std::is_same_v<ScalarA, ScalarB>) {
    if (a.data() == b.data()) {
      return true;
    }
  }
  for (int i = 0; i < isize(a); ++i) {
    if (!mochi::test::NearEqual(a[i], b[i], epsilon)) {
      return false;
    }
  }
  return true;
}

template <typename SpanA, typename SpanB>
[[nodiscard]] bool NearEqualSpan(SpanA const& a, SpanB const& b) {
  return NearEqualSpan(a, b, kDefaultNearEqualEpsilon<std::decay_t<decltype(a[0])>>);
}

/**
  Return true if the two Spans would be equal if the elements were sorted.
  Used for comparing the contents without regard to order.
*/
template <typename SpanA, typename SpanB>
[[nodiscard]] bool EqualSpanUnordered(SpanA const& a, SpanB const& b) {
  if (a.size() != b.size()) {
    return false;
  } else if (a.data() != b.data()) {
    std::vector vecA(a.begin(), a.end());
    std::vector vecB(b.begin(), b.end());
    std::sort(vecA.begin(), vecA.end());
    std::sort(vecB.begin(), vecB.end());
    return EqualSpan(vecA, vecB);
  }
  return true;
}

/**
  Return true if the matrices are the same size and if all values are NearEqual.
  This supports comparison of different sparse/dense matrix types. It is particularly fast.
*/
template <typename MatA, typename MatB, typename EpsT = typename MatA::NonConstScalar>
[[nodiscard]] inline bool NearEqualMatrices(
    MatA const& matA,
    MatB const& matB,
    EpsT epsilon = kDefaultNearEqualEpsilon<EpsT>) {
  if ((matA.Rows() != matB.Rows()) || (matA.Cols() != matB.Cols())) {
    return false;
  }
  for (int r = 0; r < matA.Rows(); ++r) {
    for (int c = 0; c < matA.Cols(); ++c) {
      if (!mochi::test::NearEqual(matA(r, c), matB(r, c), epsilon)) {
        return false;
      }
    }
  }
  return true;
}

// Get the path to the unit test's "assets" directory.
// Example: auto fullPath = GetAssetsDir() + "your_dir/your_file_name.txt";
std::string GetAssetsDir();

// Get the full path to a file in the test's "assets" directory.
// Automatically fails the test if the specified file does not exist.
// Example: auto fullPath = GetAssetPath("your_dir/your_file_name.txt");
std::string GetAssetPath(std::string const& relativePath);
std::string GetAssetPath(std::string_view relativePath);
std::string GetAssetPath(char const* relativePath);

/**
 * @brief Wraps the call to Google's testing::InitGoogleTest with some setup code that is common to
 * mochi unit tests.
 *
 * @param argc Number of arguments (mutable)
 * @param argv Array of argument strings
 *
 * @note The arguments will be forwarded to testing::InitGoogleTest, which may modify argc and argv.
 * If the calling code also wants to use argc and argv after this call, then it will need the
 * modified count. That's why argc is passed by reference.
 */
void InitUnitTest(int& argc, char** argv);

} // namespace mochi::test
