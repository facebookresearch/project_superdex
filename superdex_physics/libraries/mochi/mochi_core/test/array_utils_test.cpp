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

#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/task_scheduler.h>
#include <mochi_core/utils/transform_rt.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace mochi;

static constexpr int kTestSizes[] = {0,  1,  2,  3,  4,  5,   6,   7,   8,   9,  10, 11,
                                     12, 13, 14, 15, 16, 17,  18,  19,  20,  21, 22, 23,
                                     24, 25, 47, 48, 49, 127, 128, 129, 255, 256};
static constexpr int kLen = 256;

static void TestArrayPlusEqualsOrMinusEquals(bool add, bool aligned) {
  constexpr real kEps = 1e-5_r;
  alignas(alignof(Vec4r)) real buf1[kLen + 2];
  alignas(alignof(Vec4r)) real buf2[kLen + 2];
  real* arr1 = aligned ? buf1 : (buf1 + 1);
  real* arr2 = aligned ? buf2 : (buf2 + 1);

  // sentinel values
  arr1[kLen] = 1.23_r;
  arr2[kLen] = 4.56_r;

  // Repeat for various numbers of values
  for (int count = 0; count < kLen; ++count) {
    // Fill with values
    for (int i = 0; i < count; ++i) {
      arr1[i] = (real)(i);
      arr2[i] = (real)(2 * i);
    }

    if (add) {
      // arr1[i] += arr2[i]
      ArrayPlusEquals(Span<real>(arr1, count), Span<real const>(arr2, count));
      for (int i = 0; i < count; ++i) {
        EXPECT_NEAR(arr1[i], (real)(3 * i), kEps); // sum
        EXPECT_NEAR(arr2[i], (real)(2 * i), kEps); // unchanged
      }
    } else {
      // arr1[i] += arr2[i]
      ArrayMinusEquals(Span<real>(arr1, count), Span<real const>(arr2, count));
      for (int i = 0; i < count; ++i) {
        EXPECT_NEAR(arr1[i], (real)(-1 * i), kEps); // difference
        EXPECT_NEAR(arr2[i], (real)(2 * i), kEps); // unchanged
      }
    }

    // Check for overruns
    EXPECT_NEAR(1.23_r, arr1[kLen], kEps);
    EXPECT_NEAR(4.56_r, arr2[kLen], kEps);
  }
}

static void TestArrayPlusEqualsOrMinusEquals(bool add) {
  TestArrayPlusEqualsOrMinusEquals(add, true);
  TestArrayPlusEqualsOrMinusEquals(add, false);
}

TEST(ArrayUtils, ArrayPlusEquals) {
  // Single threaded
  TestArrayPlusEqualsOrMinusEquals(true);

  // Multithreded if a TaskScheduler is bound
  TaskScheduler scheduler(2);
  TestArrayPlusEqualsOrMinusEquals(true);
}

TEST(ArrayUtils, ArrayMinusEquals) {
  // Single threaded
  TestArrayPlusEqualsOrMinusEquals(false);

  // Multithreded if a TaskScheduler is bound
  TaskScheduler scheduler(2);
  TestArrayPlusEqualsOrMinusEquals(false);
}

static void TestArrayAddOrSub(bool add, bool aligned) {
  constexpr real kEps = 1e-5_r;
  alignas(alignof(Vec4r)) real buf1[kLen + 2];
  alignas(alignof(Vec4r)) real buf2[kLen + 2];
  alignas(alignof(Vec4r)) real buf3[kLen + 2];
  real* arr1 = aligned ? buf1 : (buf1 + 1);
  real* arr2 = aligned ? buf2 : (buf2 + 1);
  real* arr3 = aligned ? buf3 : (buf3 + 1);

  // sentinel values
  arr1[kLen] = 1.23_r;
  arr2[kLen] = 4.56_r;
  arr3[kLen] = 7.89_r;

  // Repeat for various numbers of values
  for (int count : kTestSizes) {
    EXPECT_TRUE(count <= kLen);

    Span<real> span1(arr1, count);
    Span<real const> span2(arr2, count);
    Span<real const> span3(arr3, count);

    // Fill with values
    for (int i = 0; i < count; ++i) {
      arr2[i] = (real)(2 * i);
      arr3[i] = (real)(3 * i);
    }

    if (add) {
      // arr1[i] = arr2[i] + arr3[i]
      ArrayAdd(span1, span2, span3);
      for (int i = 0; i < count; ++i) {
        EXPECT_NEAR(arr1[i], (real)(5 * i), kEps); // sum
        EXPECT_NEAR(arr2[i], (real)(2 * i), kEps); // unchanged
        EXPECT_NEAR(arr3[i], (real)(3 * i), kEps); // unchanged
      }
    } else {
      // arr1[i] = arr2[i] - arr3[i]
      ArraySub(span1, span2, span3);
      for (int i = 0; i < count; ++i) {
        EXPECT_NEAR(arr1[i], (real)(-1 * i), kEps); // difference
        EXPECT_NEAR(arr2[i], (real)(2 * i), kEps); // unchanged
        EXPECT_NEAR(arr3[i], (real)(3 * i), kEps); // unchanged
      }
    }

    // Check for overruns
    EXPECT_NEAR(1.23_r, arr1[kLen], kEps);
    EXPECT_NEAR(4.56_r, arr2[kLen], kEps);
    EXPECT_NEAR(7.89_r, arr3[kLen], kEps);
  }
}

static void TestArrayAddOrSub(bool add) {
  TestArrayAddOrSub(add, true);
  TestArrayAddOrSub(add, false);
}

static void TestArrayInverts(bool aligned) {
  constexpr real kEps = 1e-5_r;
  alignas(alignof(Vec4r)) real buf1[kLen + 2];
  real* arr1 = aligned ? buf1 : (buf1 + 1);

  // sentinel values
  arr1[kLen] = 1.23_r;

  // Repeat for various numbers of values
  for (int count : kTestSizes) {
    EXPECT_TRUE(count <= kLen);

    // Fill with values
    for (int i = 0; i < count; ++i) {
      arr1[i] = (real)(2 * (i + 1));
    }

    // arr1[i] *= arr2[i]
    ArrayInverts(Span<real>(arr1, count));
    for (int i = 0; i < count; ++i) {
      EXPECT_NEAR(arr1[i], (real)(1) / real(2 * (i + 1)), kEps);
    }

    // Check for overruns
    EXPECT_NEAR(1.23_r, arr1[kLen], kEps);
  }
}

static void TestArrayScales(bool aligned) {
  constexpr real kEps = 1e-5_r;
  alignas(alignof(Vec4r)) real buf1[kLen + 2];
  real* arr1 = aligned ? buf1 : (buf1 + 1);

  // sentinel values
  arr1[kLen] = 1.23_r;

  // scaling factor
  auto alpha = static_cast<real>(1.414213562);

  // Repeat for various numbers of values
  for (int count : kTestSizes) {
    EXPECT_TRUE(count <= kLen);

    // Fill with values
    for (int i = 0; i < count; ++i) {
      arr1[i] = (real)(i + 1);
    }

    // arr1[i] *= alpha
    ArrayMulEquals(Span<real>(arr1, count), alpha);
    for (int i = 0; i < count; ++i) {
      EXPECT_NEAR(arr1[i], (real)(i + 1) * alpha, kEps);
    }

    // Check for overruns
    EXPECT_NEAR(1.23_r, arr1[kLen], kEps);
  }
}

static void TestArrayMul(bool aligned) {
  constexpr real kEps = 1e-5_r;
  alignas(alignof(Vec4r)) real buf1[kLen + 2];
  alignas(alignof(Vec4r)) real buf2[kLen + 2];
  real* arr1 = aligned ? buf1 : (buf1 + 1);
  real* arr2 = aligned ? buf2 : (buf2 + 1);

  // sentinel values
  arr1[kLen] = 1.23_r;
  arr2[kLen] = 4.56_r;

  // Repeat for various numbers of values
  for (int count : kTestSizes) {
    EXPECT_TRUE(count <= kLen);

    // Fill with values
    for (int i = 0; i < count; ++i) {
      arr1[i] = (real)(2 * (i + 1));
      arr2[i] = (real)(3 * (i + 1));
    }

    // arr1[i] *= arr2[i]
    ArrayMulEquals(Span<real>(arr1, count), Span<real const>(arr2, count));
    for (int i = 0; i < count; ++i) {
      EXPECT_NEAR(arr1[i], (real)(6 * (i + 1) * (i + 1)), kEps);
      EXPECT_NEAR(arr2[i], (real)(3 * (i + 1)), kEps); // unchanged
    }

    // Check for overruns
    EXPECT_NEAR(1.23_r, arr1[kLen], kEps);
    EXPECT_NEAR(4.56_r, arr2[kLen], kEps);
  }
}

static void TestArrayAddCoord(bool aligned) {
  alignas(alignof(Vec4r)) Real3 buf1[kLen + 2];
  alignas(alignof(Vec4r)) Real3 buf2[kLen + 2];
  Real3* arr1 = aligned ? buf1 : (buf1 + 1);
  Real3* arr2 = aligned ? buf2 : (buf2 + 1);

  // sentinel values
  Real3 sentinel = Real3{1.23_r, 2.34_r, 3.45_r};
  arr1[kLen] = sentinel;

  // Repeat for various numbers of values
  for (int numCoords : kTestSizes) {
    EXPECT_TRUE(numCoords <= kLen);

    // Fill with values
    for (int i = 0; i < numCoords; ++i) {
      arr2[i][0] = (real)(2 * i);
      arr2[i][1] = (real)(3 * i);
      arr2[i][2] = (real)(4 * i);
    }

    arr1[numCoords] = sentinel;
    arr2[numCoords] = sentinel;

    // arr1[i] = arr2[i] + coord
    Real3 coord{1.1_r, 2.2_r, 3.3_r};
    ArrayAdd(Span{arr1, size_t(numCoords)}, Span<Real3 const>{arr2, size_t(numCoords)}, coord);
    for (int i = 0; i < numCoords; ++i) {
      EXPECT_NEAR_EQ(arr1[i], arr2[i] + coord);
      EXPECT_NEAR_EQ(arr2[i], Real3(2_r * i, 3_r * i, 4_r * i)); // unchanged
    }

    // Check for overruns
    EXPECT_NEAR_EQ(sentinel, arr1[numCoords]);
    EXPECT_NEAR_EQ(sentinel, arr2[numCoords]);

    // arr1[i] = arr1[i] - coord
    ArrayAdd(Span{arr1, size_t(numCoords)}, Span<Real3 const>{arr1, size_t(numCoords)}, -coord);
    for (int i = 0; i < numCoords; ++i) {
      EXPECT_TRUE(NearEqual(arr1[i], arr2[i], 5e-5_r));
    }

    // Check for overruns
    EXPECT_NEAR_EQ(sentinel, arr1[numCoords]);
    EXPECT_NEAR_EQ(sentinel, arr2[numCoords]);
  }
  EXPECT_NEAR_EQ(sentinel, arr1[kLen]);
}

static void TestArrayAddCoord() {
  TestArrayAddCoord(true);
  TestArrayAddCoord(false);
}

TEST(ArrayUtils, ArrayAdd) {
  // Single threaded
  TestArrayAddOrSub(true);
  TestArrayAddCoord();

  // Multithreded if a TaskScheduler is bound
  TaskScheduler scheduler(2);
  TestArrayAddOrSub(true);
  TestArrayAddCoord();
}

TEST(ArrayUtils, ArraySub) {
  // Single threaded
  TestArrayAddOrSub(false);

  // Multithreaded if a TaskScheduler is bound
  TaskScheduler scheduler(2);
  TestArrayAddOrSub(false);
}

TEST(ArrayUtils, ArrayMul) {
  TestArrayMul(true);
  TestArrayMul(false);

  // Multithreaded if a TaskScheduler is bound
  TaskScheduler scheduler(2);
  TestArrayMul(true);
  TestArrayMul(false);
}

TEST(ArrayUtils, ArrayInverts) {
  TestArrayInverts(true);
  TestArrayInverts(false);

  // Multithreaded if a TaskScheduler is bound
  TaskScheduler scheduler(2);
  TestArrayInverts(true);
  TestArrayInverts(false);
}

TEST(ArrayUtils, ArrayScales) {
  TestArrayScales(true);
  TestArrayScales(false);

  // Multithreaded if a TaskScheduler is bound
  TaskScheduler scheduler(2);
  TestArrayScales(true);
  TestArrayScales(false);
}

template <bool kSingleThreaded>
static void TestArrayTransformPoints() {
  // Generate several arbitrary points
  int constexpr kNumPoints = 100;
  std::vector<Real3> points;
  points.resize(kNumPoints);
  for (int i = 0; i < kNumPoints; ++i) {
    points[i] = Real3{(real)(i % 7), -1.23_r * (real)(i % 13), 2.34_r * (real)(i % 19)};
  }

  // Generate an arbitrary transform with both rotation and translation
  TransformRT const transform{
      Quaternion::FromAxisAngle(Normalize(Real3{1_r, 2_r, 3_r}), kPI / 4_r),
      Real3{0.1_r, 0.2_r, 0.3_r}};

  Real3 const sentinel{911_r, 911_r, 911_r};
  real constexpr kEpsilon = 1e-5_r;

  // Helper function verifies that:
  //  results[i] == transform.TransformPoint(points[i]); // whatever that may be
  auto expectResults = [&](Span<Real3 const> results, int count) {
    EXPECT_LE(count, isize(points));
    EXPECT_LE(count, isize(results));
    for (int i = 0; i < count; ++i) {
      Real3 expectedResult = transform.TransformPoint(points[i]);
      EXPECT_TRUE(NearEqual(expectedResult, results[i], kEpsilon));
    }
  };

  // zero points
  {
    std::vector<Real3> results;
    ArrayTransformPoints<kSingleThreaded>(
        MakeSpan(results), Span<Real3 const>{points.data(), 0}, transform); // don't crash
  }

  // 1 point
  {
    std::vector<Real3> results;
    results.resize(2);
    results[1] = sentinel;
    ArrayTransformPoints<kSingleThreaded>(
        Span<Real3>{results.data(), 1}, Span<Real3 const>{points.data(), 1}, transform);
    expectResults(results, 1);
    EXPECT_NEAR_EQ(sentinel, results[1]); // expect no change
  }

  // many points
  for (size_t count = 1; count < points.size(); ++count) {
    std::vector<Real3> results;
    results.resize(count + 1, sentinel);
    ArrayTransformPoints<kSingleThreaded>(
        Span<Real3>{results.data(), count}, Span<Real3 const>{points.data(), count}, transform);
    expectResults(results, (int)count);
    EXPECT_NEAR_EQ(sentinel, results[count]); // expect no change
  }

  // in-place
  {
    std::vector<Real3> results = points;
    ArrayTransformPoints<kSingleThreaded>(
        Span<Real3>{results}, Span<Real3 const>{results}, transform);
    expectResults(results, isize(points));
  }
}

TEST(ArrayUtils, ArrayTransformPoints) {
  TestArrayTransformPoints<false>();
  TestArrayTransformPoints<true>();
}

template <bool kSingleThreaded>
static void TestArrayTransformDisplacements() {
  // Generate several arbitrary displacements
  int constexpr kNumPoints = 100;
  std::vector<Real3> displacements;
  displacements.resize(kNumPoints);
  for (int i = 0; i < kNumPoints; ++i) {
    displacements[i] = Real3{(real)(i % 7), -1.23_r * (real)(i % 13), 2.34_r * (real)(i % 19)};
  }

  // Generate an equal number of arbitrary coordinates (our reference mesh)
  std::vector<Real3> refCoords;
  refCoords.resize(kNumPoints);
  for (int i = 0; i < kNumPoints; ++i) {
    refCoords[i] = Real3{(real)(i % 9), 1.45_r * (real)(i % 27), -2.56_r * (real)(i % 31)};
  }

  // Generate an arbitrary transform with both rotation and translation
  TransformRT const transform{
      Quaternion::FromAxisAngle(Normalize(Real3{1_r, 2_r, 3_r}), kPI / 4_r),
      Real3{0.1_r, 0.2_r, 0.3_r}};

  Real3 const sentinel{911_r, 911_r, 911_r};
  real constexpr kEpsilon = 1e-5_r;

  // Helper function verifies that:
  //  results[i] == (transform.TransformPoint(refCooords[i] + displacements[i]) - refCoords[i])
  auto expectResults = [&](Span<Real3 const> results, int count) {
    EXPECT_LE(count, isize(displacements));
    EXPECT_LE(count, isize(results));
    EXPECT_LE(count, isize(refCoords));
    for (int i = 0; i < count; ++i) {
      Real3 expectedResult =
          transform.TransformPoint(refCoords[i] + displacements[i]) - refCoords[i];
      EXPECT_TRUE(NearEqual(expectedResult, results[i], kEpsilon));
    }
  };

  // zero points
  {
    std::vector<Real3> results;
    ArrayTransformDisplacements<kSingleThreaded>(
        Span<Real3>{results},
        Span<Real3 const>{displacements.data(), 0},
        Span<Real3 const>{refCoords.data(), 0},
        transform); // don't crash
  }

  // 1 point
  {
    std::vector<Real3> results;
    results.resize(2);
    results[1] = sentinel;
    ArrayTransformDisplacements<kSingleThreaded>(
        Span<Real3>{results.data(), 1},
        Span<Real3 const>{displacements.data(), 1},
        Span<Real3 const>{refCoords.data(), 1},
        transform);
    expectResults(results, 1);
    EXPECT_NEAR_EQ(sentinel, results[1]); // expect no change
  }

  // many points
  for (size_t count = 1; count < displacements.size(); ++count) {
    std::vector<Real3> results;
    results.resize(count + 1, sentinel);
    ArrayTransformDisplacements<kSingleThreaded>(
        Span<Real3>{results.data(), count},
        Span<Real3 const>{displacements.data(), count},
        Span<Real3 const>{refCoords.data(), count},
        transform);
    expectResults(results, (int)count);
    EXPECT_NEAR_EQ(sentinel, results[count]); // expect no change
  }

  // in-place
  {
    std::vector<Real3> results = displacements;
    ArrayTransformDisplacements<kSingleThreaded>(
        Span<Real3>{results}, Span<Real3 const>{results}, Span<Real3 const>{refCoords}, transform);
    expectResults(results, isize(displacements));
  }
}

TEST(ArrayUtils, ArrayTransformDisplacements) {
  TestArrayTransformDisplacements<false>();
  TestArrayTransformDisplacements<true>();
}

template <bool kSingleThreaded>
static void TestArrayRotateVectors() {
  // Generate several arbitrary points
  int constexpr kNumPoints = 100;
  std::vector<Real3> points;
  points.resize(kNumPoints);
  for (int i = 0; i < kNumPoints; ++i) {
    points[i] = Real3{(real)(i % 7), -1.23_r * (real)(i % 13), 2.34_r * (real)(i % 19)};
  }

  // Generate an arbitrary transform with both rotation and translation
  Quaternion const rotation = Quaternion::FromAxisAngle(Normalize(Real3{1_r, 2_r, 3_r}), kPI / 4_r);
  Real3 const sentinel{911_r, 911_r, 911_r};
  real constexpr kEpsilon = 1e-5_r;

  // Helper function verifies that:
  //  results[i] == rotation * points[i]; // whatever that may be
  auto expectResults = [&](Span<Real3 const> results, int count) {
    EXPECT_LE(count, isize(points));
    EXPECT_LE(count, isize(results));
    for (int i = 0; i < count; ++i) {
      Real3 expectedResult = rotation * points[i];
      EXPECT_TRUE(NearEqual(expectedResult, results[i], kEpsilon));
    }
  };

  // zero points
  {
    std::vector<Real3> results;
    ArrayRotateVectors<kSingleThreaded>(
        Span<Real3>{results}, Span<Real3 const>{points.data(), 0}, rotation); // don't crash
  }

  // 1 point
  {
    std::vector<Real3> results;
    results.resize(2);
    results[1] = sentinel;
    ArrayRotateVectors<kSingleThreaded>(
        Span<Real3>{results.data(), 1}, Span<Real3 const>{points.data(), 1}, rotation);
    expectResults(results, 1);
    EXPECT_NEAR_EQ(sentinel, results[1]); // expect no change
  }

  // many points
  for (size_t count = 1; count < points.size(); ++count) {
    std::vector<Real3> results;
    results.resize(count + 1, sentinel);
    ArrayRotateVectors<kSingleThreaded>(
        Span<Real3>{results.data(), count}, Span<Real3 const>{points.data(), count}, rotation);
    expectResults(results, (int)count);
    EXPECT_NEAR_EQ(sentinel, results[count]); // expect no change
  }

  // in-place
  {
    std::vector<Real3> results = points;
    ArrayRotateVectors<kSingleThreaded>(Span<Real3>{results}, Span<Real3 const>{results}, rotation);
    expectResults(results, isize(points));
  }
}

TEST(ArrayUtils, ArrayRotateVectors) {
  TestArrayRotateVectors<false>();
  TestArrayRotateVectors<true>();
}

TEST(ArrayUtils, MinMax) {
  constexpr std::array<real, 21> kValues = {
      -5, 6,   -7, -1, 2,  0,   -9, 8, 12,  -13, 4,
      10, -11, -3, 17, 21, -14, 0,  1, -21, 31}; // jumbled order

  auto computeExpectedMinMax = [](Span<real const> span) -> std::pair<real, real> {
    if (span.empty()) {
      return {0_r, 0_r};
    }
    real minVal = span[0];
    real maxVal = span[0];
    for (size_t i = 1; i < span.size(); ++i) {
      minVal = std::min(minVal, span[i]);
      maxVal = std::max(maxVal, span[i]);
    }
    return {minVal, maxVal};
  };

  for (size_t startIdx = 0; startIdx <= 1; ++startIdx) {
    for (size_t length = 1; length <= kValues.size() - startIdx; ++length) {
      Span<real const> subspan{&kValues[startIdx], length};

      auto [expectedMin, expectedMax] = computeExpectedMinMax(subspan);

      real computedMin = Min(subspan);
      real computedMax = Max(subspan);
      auto [computedMinFromMinMax, computedMaxFromMinMax] = MinMax(subspan);

      EXPECT_EQ(expectedMin, computedMin);
      EXPECT_EQ(expectedMin, computedMinFromMinMax);
      EXPECT_EQ(expectedMax, computedMax);
      EXPECT_EQ(expectedMax, computedMaxFromMinMax);
    }
  }
}

TEST(ArrayUtils, MaxAbs) {
  for (int sz : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 31, 32, 33}) {
    std::vector<real> vec(sz, 0_r);
    EXPECT_EQ(0_r, MaxAbs(MakeConstSpan(vec)));
    for (int i = 0; i < sz; ++i) {
      for (int j = 0; j < sz; ++j) {
        if (i != j) {
          auto previ = vec[i];
          auto prevj = vec[j];
          // Introduce a small difference at index i
          vec[i] -= 0.1_r;
          EXPECT_NEAR_EQ(0.1_r, MaxAbs(MakeConstSpan(vec)));
          // Introduce a larger difference at index j
          vec[j] += 0.2_r;
          EXPECT_NEAR_EQ(0.2_r, MaxAbs(MakeConstSpan(vec)));
          // Revert
          vec[i] = previ;
          vec[j] = prevj;
          EXPECT_EQ(0_r, MaxAbs(MakeConstSpan(vec)));
        }
      }
    }
  }
}

TEST(ArrayUtils, MaxAbsDifference) {
  for (int sz : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 31, 32, 33}) {
    std::vector<real> vecA(sz);
    std::vector<real> vecB(sz);
    for (int i = 0; i < sz; ++i) {
      vecA[i] = vecB[i] = static_cast<real>(i); // arbitrary values
    }
    auto spanA = MakeConstSpan(vecA);
    auto spanB = MakeConstSpan(vecB);
    EXPECT_EQ(0_r, MaxAbsDifference(spanA, spanB));
    EXPECT_EQ(0_r, MaxAbsDifference(spanB, spanA));
    for (int i = 0; i < sz; ++i) {
      for (int j = 0; j < sz; ++j) {
        if (i != j) {
          // Introduce a small difference at index i
          vecA[i] -= 0.1_r;
          EXPECT_NEAR_EQ(0.1_r, MaxAbsDifference(spanA, spanB));
          EXPECT_NEAR_EQ(0.1_r, MaxAbsDifference(spanB, spanA));
          // Introduce a larger difference at index j
          vecA[j] += 0.2_r;
          EXPECT_NEAR_EQ(0.2_r, MaxAbsDifference(spanA, spanB));
          EXPECT_NEAR_EQ(0.2_r, MaxAbsDifference(spanB, spanA));
          // Revert
          vecA[i] = vecB[i];
          vecA[j] = vecB[j];
          EXPECT_EQ(0_r, MaxAbsDifference(spanA, spanB));
        }
      }
    }
  }
}

TEST(ArrayUtils, HSum) {
  ColumnVector<real> values(256);
  values.SetRandom(123, -1_r, 1_r);
  real expectedSum = 0_r;
  EXPECT_NEAR_EQ(expectedSum, HSum(Span<real const>{})); // empty span
  for (size_t i = 0; i < values.size(); ++i) {
    expectedSum += values[i];
    auto reportedSum = HSum(Span<real const>{values.data(), i + 1});
    auto tolerance = kDefaultNearEqualEpsilon<real> * i; // Increases with i
    EXPECT_NEAR_TOL(expectedSum, reportedSum, tolerance);
  }
}

TEST(ArrayUtils, ArgSort) {
  {
    // Already sorted array
    std::vector<int> data = {1, 2, 3, 4, 5};
    auto result = ArgSort(MakeConstSpan(data));
    EXPECT_SPAN_EQ(result, (std::vector<size_t>{0, 1, 2, 3, 4}));
  }

  {
    // Reverse sorted array
    std::vector<int> data = {5, 4, 3, 2, 1};
    auto result = ArgSort(MakeConstSpan(data));
    EXPECT_SPAN_EQ(result, (std::vector<size_t>{4, 3, 2, 1, 0}));
  }

  {
    // Array with negative numbers
    std::vector<int> data = {-10, 0, 5, -3, 2};
    auto result = ArgSort(MakeConstSpan(data));
    EXPECT_SPAN_EQ(result, (std::vector<size_t>{0, 3, 1, 4, 2}));
  }

  {
    // Empty array
    std::vector<int> data = {};
    auto result = ArgSort(MakeConstSpan(data));
    EXPECT_TRUE(result.empty());
  }

  {
    // Single-element array
    std::vector<int> data = {42};
    auto result = ArgSort(MakeConstSpan(data));
    EXPECT_SPAN_EQ(result, (std::vector<size_t>{0}));
  }

  {
    // Large randomly shuffled array
    std::vector<int> data(100000);
    std::iota(data.begin(), data.end(), 0);
    std::mt19937 rng(42); // Fixed seed for reproducibility
    std::shuffle(data.begin(), data.end(), rng);

    auto result = ArgSort(MakeConstSpan(data));

    // Reconstruct the data in sorted order using argsort result
    // This should yield a sorted sequence from the shuffled input
    std::vector<int> sorted;
    sorted.reserve(data.size());
    for (auto idx : result) {
      sorted.push_back(data[idx]);
    }
    EXPECT_TRUE(std::is_sorted(sorted.begin(), sorted.end()));

    // Check that result is a valid permutation: no duplicates or missing indices
    std::vector<size_t> expected(result.size());
    std::iota(expected.begin(), expected.end(), 0);
    std::sort(result.begin(), result.end());
    EXPECT_SPAN_EQ(result, expected);
  }
}

TEST(ArrayUtils, SortAndRemoveDuplicates) {
  {
    DynamicArray<int> v{1, 2, 3, 4};
    DynamicArray<int> const expected{1, 2, 3, 4};
    SortAndRemoveDuplicates(v);
    EXPECT_SPAN_EQ(v, expected);
  }

  {
    DynamicArray<int> v{3, 1, 2, 2, 3, 1};
    DynamicArray<int> const expected{1, 2, 3};
    SortAndRemoveDuplicates(v);
    EXPECT_SPAN_EQ(v, expected);
  }

  {
    DynamicArray<int> v{7, 7, 7, 7, 7};
    DynamicArray<int> const expected{7};
    SortAndRemoveDuplicates(v);
    EXPECT_SPAN_EQ(v, expected);
  }

  {
    DynamicArray<int> v{};
    DynamicArray<int> const expected{};
    SortAndRemoveDuplicates(v);
    EXPECT_SPAN_EQ(v, expected);
  }

  {
    DynamicArray<std::string> v{"b", "a", "a", "c", "b"};
    DynamicArray<std::string> const expected{"a", "b", "c"};
    SortAndRemoveDuplicates(v);
    EXPECT_SPAN_EQ(v, expected);
  }
}

TEST(ArrayUtils, IsFinite) {
  constexpr size_t kMaxTestSize = 5 * Simd<real>::kSize + /* ±min, ±max, ±lowest */ 6;

  // Empty span.
  DynamicArray<real> emptyArray;
  EXPECT_TRUE(IsFinite(MakeSpan(emptyArray)));

  // Test a type with SIMD support using different sizes and different non-finite values at
  // different indices.
  DynamicArray<real> array;
  array.reserve(kMaxTestSize);
  array.push_back(std::numeric_limits<real>::min());
  array.push_back(-std::numeric_limits<real>::min());
  array.push_back(std::numeric_limits<real>::max());
  array.push_back(-std::numeric_limits<real>::max());
  array.push_back(std::numeric_limits<real>::lowest());
  array.push_back(-std::numeric_limits<real>::lowest());
  while (array.size() < kMaxTestSize) {
    array.push_back(static_cast<real>(array.size()));
  }
  EXPECT_TRUE(IsFinite(MakeSpan(array)));
  EXPECT_TRUE(IsFinite(MakeConstSpan(array))); // Also const overload

  for (int i = 0; i < isize(array); ++i) {
    for (int j = 0; j < i; ++j) {
      real const prevValue = array[j];

      auto testArray = array;
      testArray[j] = -std::numeric_limits<real>::infinity();
      EXPECT_FALSE(IsFinite(MakeSpan(testArray)));

      testArray[j] = prevValue;
      EXPECT_TRUE(IsFinite(MakeSpan(testArray)));

      testArray[j] = std::numeric_limits<real>::signaling_NaN();
      EXPECT_FALSE(IsFinite(MakeSpan(testArray)));

      testArray[j] = prevValue;
      EXPECT_TRUE(IsFinite(MakeSpan(testArray)));

      testArray[j] = std::numeric_limits<real>::quiet_NaN();
      EXPECT_FALSE(IsFinite(MakeSpan(testArray)));
    }
  }

  // Test a type without SIMD support.
  DynamicArray<TransformRT> transformArray(kMaxTestSize, TransformRT{});
  EXPECT_TRUE(IsFinite(MakeSpan(transformArray)));
  transformArray[0].SetTranslation(Real3{0_r, 0_r, std::numeric_limits<real>::infinity()});
  EXPECT_FALSE(IsFinite(MakeSpan(transformArray)));
}

TEST(ArrayUtils, Fill) {
  auto testFill = [](size_t sz, auto const& value, auto const& sentinel) {
    using T = std::remove_const_t<std::decay_t<decltype(value)>>;
    DynamicArray<T> arr(sz + 1, sentinel);
    auto span = Span{arr.data(), sz};
    Fill(span, value);
    for (auto const& x : span) {
      EXPECT_EQ(value, x);
    }
    EXPECT_EQ(sentinel, arr[sz]); // no change
  };

  for (size_t sz = 0; sz < 20; ++sz) {
    // real (4 or 8 bytes) uses std::fill
    testFill(sz, 42_r, 911_r);

    // int (4 bytes) uses std::fill
    testFill(sz, 42, 911);

    // NdArray<real, 3> is 12 or 24 bytes, uses NdArray specialization
    testFill(sz, Real3{1_r, 2_r, 3_r}, Real3{911_r, 911_r, 911_r});

    // NdArray<int, 4> is 16 bytes, uses NdArray specialization
    testFill(sz, Int4{1, 2, 3, 4}, Int4{911, 911, 911, 911});

    // std::array<uint8_t, 3> is 3 bytes, uses std::fill
    testFill(sz, std::array<uint8_t, 3>{1, 2, 3}, std::array<uint8_t, 3>{42, 42, 42});

    // std::array<uint8_t, 4> is 4 bytes, uses specialization for trivially copyable types
    testFill(sz, std::array<uint8_t, 4>{1, 2, 3, 4}, std::array<uint8_t, 4>{42, 42, 42, 42});

    // std::array<real, 6> is 24 or 48 bytes, uses specialization for trivially copyable types
    testFill(
        sz,
        std::array<real, 6>{1_r, 2_r, 3_r, 4_r, 5_r, 6_r},
        std::array<real, 6>{911_r, 911_r, 911_r, 911_r, 911_r, 911_r});

    // std::string is not trivially copyable, uses std::fill
    testFill(sz, std::string("woot"), std::string("nope"));
  }
}
