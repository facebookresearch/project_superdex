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

#include <mochi_core/linear_algebra/matrix_fwd.h>
#include <mochi_core/linear_algebra/matrix_operations.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/matrix_utils.h>
#include <mochi_core/utils/rand_utils.h>

#include <gtest/gtest.h>

#include <vector>

using namespace mochi;

TEST(MatrixUtils, NearEqual) {
  // Matrix2x3r
  {
    constexpr Matrix2x3r a = {Real3{1_r, 2_r, 3_r}, Real3{4_r, 5_r, 6_r}};
    constexpr Matrix2x3r b = {Real3{1_r, 2_r, 3_r}, Real3{4_r, 5_r, 6_r}};
    constexpr Matrix2x3r c = {Real3{1_r, 2_r, 3_r}, Real3{4_r, 5_r, 6.0000001_r}};
    constexpr Matrix2x3r d = {Real3{1_r, 2_r, 3_r}, Real3{4_r, 5_r, 6.000002_r}};
    static_assert(NearEqual(a, a));
    static_assert(NearEqual(a, b));
    static_assert(NearEqual(a, c, 1e-6_r));
    static_assert(!NearEqual(a, d, 1e-6_r));
  }
}

TEST(MatrixUtils, Transpose) {
  // 2x2
  {
    constexpr Matrix2x2r m = {
        Real2{1_r, 2_r}, //
        Real2{3_r, 4_r}};

    constexpr Matrix2x2r t = {
        Real2{1_r, 3_r}, //
        Real2{2_r, 4_r}};

    static_assert(t == Transpose(m));
    static_assert(m == Transpose(Transpose(m)));
  }

  // 3x2
  {
    constexpr Matrix3x2r m = {
        Real2{1_r, 2_r}, //
        Real2{3_r, 4_r}, //
        Real2{5_r, 6_r}};

    constexpr Matrix2x3r t = {
        Real3{1_r, 3_r, 5_r}, //
        Real3{2_r, 4_r, 6_r}};

    static_assert(t == Transpose(m));
    static_assert(m == Transpose(Transpose(m)));
  }

  // 2x2 SIMD
  {
    // Input row-major: [a00, a01, a10, a11] = [1, 2, 3, 4]
    Vec4r const M(1_r, 2_r, 3_r, 4_r);
    // Expected transpose: [a00, a10, a01, a11] = [1, 3, 2, 4]
    Vec4r const result = Transpose2x2(M);
    EXPECT_EQ(Vec4r(1_r, 3_r, 2_r, 4_r), result);
  }

  // 3x3 SIMD
  {
    NdArray<Vec4r, 3> const M = {
        Vec4r(1_r, 2_r, 3_r, 0_r), //
        Vec4r(5_r, 6_r, 7_r, 0_r), //
        Vec4r(9_r, 10_r, 11_r, 0_r)};
    auto result = Transpose3x3(M);
    EXPECT_NEAR_EQ(Vec4r(1_r, 5_r, 9.0_r, 0_r), result[0]);
    EXPECT_NEAR_EQ(Vec4r(2_r, 6_r, 10_r, 0_r), result[1]);
    EXPECT_NEAR_EQ(Vec4r(3_r, 7_r, 11_r, 0_r), result[2]);
  }

  // 3x3 SIMD given a 4x4 input matrix
  {
    NdArray<Vec4r, 4> const M = {
        Vec4r(1_r, 2_r, 3_r, 0_r),
        Vec4r(5_r, 6_r, 7_r, 0_r),
        Vec4r(9_r, 10_r, 11_r, 0_r),
        Vec4r(911_r, 911_r, 911_r, 911_r) // This row should be allowed, but ignored.
    };
    VMatrix3x3r result = Transpose3x3(M);
    EXPECT_NEAR_EQ(Vec4r(1_r, 5_r, 9.0_r, 0_r), result[0]);
    EXPECT_NEAR_EQ(Vec4r(2_r, 6_r, 10_r, 0_r), result[1]);
    EXPECT_NEAR_EQ(Vec4r(3_r, 7_r, 11_r, 0_r), result[2]);
  }

  // 4x4 SIMD
  {
    NdArray<Vec4r, 4> const M = {
        Vec4r(1_r, 2_r, 3_r, 4_r),
        Vec4r(5_r, 6_r, 7_r, 8_r),
        Vec4r(9_r, 10_r, 11_r, 12_r),
        Vec4r(13_r, 14_r, 15_r, 16_r)};
    auto result = Transpose4x4(M);
    EXPECT_NEAR_EQ(Vec4r(1_r, 5_r, 9_r, 13_r), result[0]);
    EXPECT_NEAR_EQ(Vec4r(2_r, 6_r, 10_r, 14_r), result[1]);
    EXPECT_NEAR_EQ(Vec4r(3_r, 7_r, 11_r, 15_r), result[2]);
    EXPECT_NEAR_EQ(Vec4r(4_r, 8_r, 12_r, 16_r), result[3]);
  }
}

TEST(MatrixUtils, Dot) {
  // Dot(Matrix3x3r, Matrix3x3r)
  {
    constexpr Matrix3x3r a = {
        //
        Real3{1_r, 2_r, 3_r},
        Real3{4_r, 5_r, 6_r},
        Real3{7_r, 8_r, 9_r}};
    constexpr Matrix3x3r b = {
        //
        Real3{11_r, 22_r, 33_r},
        Real3{44_r, 55_r, 66_r},
        Real3{77_r, 88_r, 99_r}};
    constexpr Matrix3x3r expected = {
        //
        Real3{330_r, 396_r, 462_r},
        Real3{726_r, 891_r, 1056_r},
        Real3{1122_r, 1386_r, 1650_r}};
    static_assert(NearEqual(expected, Dot(a, b)));
  }

  // Dot(Matrix4x4r, Matrix4x4r)
  {
    constexpr Matrix4x4r a = {
        //
        Real4{1_r, 2_r, 3_r, 4_r},
        Real4{5_r, 6_r, 7_r, 8_r},
        Real4{9_r, 10_r, 11_r, 12_r},
        Real4{13_r, 14_r, 15_r, 16_r}};
    constexpr Matrix4x4r b = {
        //
        Real4{10_r, 20_r, 30_r, 40_r},
        Real4{50_r, 60_r, 70_r, 80_r},
        Real4{90_r, 100_r, 110_r, 120_r},
        Real4{130_r, 140_r, 150_r, 160_r}};
    constexpr Matrix4x4r expected = {
        //
        Real4{900_r, 1000_r, 1100_r, 1200_r},
        Real4{2020_r, 2280_r, 2540_r, 2800_r},
        Real4{3140_r, 3560_r, 3980_r, 4400_r},
        Real4{4260_r, 4840_r, 5420_r, 6000_r}};
    static_assert(NearEqual(expected, Dot(a, b)));
  }

  // Dot(VMatrix4x4r, VMatrix4x4r)
  {
    VMatrix4x4r a = {
        //
        Vec4r(1_r, 2_r, 3_r, 4_r),
        Vec4r(5_r, 6_r, 7_r, 8_r),
        Vec4r(9_r, 10_r, 11_r, 12_r),
        Vec4r(13_r, 14_r, 15_r, 16_r)};
    VMatrix4x4r b = {
        //
        Vec4r(10_r, 20_r, 30_r, 40_r),
        Vec4r(50_r, 60_r, 70_r, 80_r),
        Vec4r(90_r, 100_r, 110_r, 120_r),
        Vec4r(130_r, 140_r, 150_r, 160_r)};
    VMatrix4x4r expected = {
        //
        Vec4r(900_r, 1000_r, 1100_r, 1200_r),
        Vec4r(2020_r, 2280_r, 2540_r, 2800_r),
        Vec4r(3140_r, 3560_r, 3980_r, 4400_r),
        Vec4r(4260_r, 4840_r, 5420_r, 6000_r)};
    auto actual = Dot4x4(a, b);
    EXPECT_NEAR_EQ(expected, actual);
  }
}

TEST(MatrixUtils, DotVecMat) {
  // DotVecMat(Real3, Matrix2x3r)
  {
    constexpr Real2 a = {1_r, 2_r};
    constexpr Matrix2x3r b = {Real3{3_r, 4_r, 5_r}, Real3{6_r, 7_r, 8_r}};
    static_assert(NearEqual(Real3{15_r, 18_r, 21_r}, DotVecMat(a, b)));
  }

  // DotVecMat(Vec4r, VMatrix2x3r). 2x3 SIMD version.
  {
    Vec4r a = Vec4r(1_r, 2_r);
    VMatrix2x3r b = {Vec4r(3_r, 4_r, 5_r), Vec4r(6_r, 7_r, 8_r)};
    EXPECT_NEAR_EQ(Vec4r(15_r, 18_r, 21_r), ToSimdDirection(DotVecMat2x3(a, b)));
  }

  // DotVecMat(Real3, Matrix3x2r)
  {
    constexpr Real3 a = {1_r, 2_r, 3_r};
    constexpr Matrix3x2r b = {Real2{4_r, 5_r}, Real2{6_r, 7_r}, Real2{8_r, 9_r}};
    static_assert(NearEqual(Real2{40_r, 46_r}, DotVecMat(a, b)));
  }

  // DotVecMat(Real3, Matrix3x3r)
  {
    constexpr Real3 a = {1_r, 2_r, 3_r};
    constexpr Matrix3x3r b = {Real3{4_r, 5_r, 6_r}, Real3{7_r, 8_r, 9_r}, Real3{10_r, 11_r, 12_r}};
    static_assert(NearEqual(Real3{48_r, 54_r, 60_r}, DotVecMat(a, b)));
  }

  // DotVecMat(Vec4r, VMatrix3x3r). 3x3 SIMD version.
  {
    Vec4r a(1_r, 2_r, 3_r, 111_r);
    VMatrix3x3r b = {
        Vec4r(4_r, 5_r, 6_r, 222_r), Vec4r(7_r, 8_r, 9_r, 333_r), Vec4r(10_r, 11_r, 12_r, 444_r)};
    EXPECT_NEAR_EQ(Vec4r(48_r, 54_r, 60_r), ToSimdDirection(DotVecMat3x3(a, b)));
  }

  // DotVecMat(Vec4r, VMatrix4x4r). Only uses the upper-left 3x3 portion.
  {
    Vec4r a(1_r, 2_r, 3_r, 111_r);
    VMatrix4x4r b = {
        Vec4r(4_r, 5_r, 6_r, 222_r),
        Vec4r(7_r, 8_r, 9_r, 333_r),
        Vec4r(10_r, 11_r, 12_r, 444_r),
        Vec4r(13_r, 14_r, 15_r, 555_r)};
    EXPECT_NEAR_EQ(Vec4r(48_r, 54_r, 60_r), ToSimdDirection(DotVecMat3x3(a, b)));
  }

  // DotVecMat(Real4, Matrix4x4r)
  {
    constexpr Real4 a = {1_r, 2_r, 3_r, 4_r};
    constexpr Matrix4x4r b = {
        Real4{10_r, 20_r, 30_r, 40_r},
        Real4{50_r, 60_r, 70_r, 80_r},
        Real4{90_r, 100_r, 110_r, 120_r},
        Real4{130_r, 140_r, 150_r, 160_r}};
    static_assert(NearEqual(Real4{900_r, 1000_r, 1100_r, 1200_r}, DotVecMat(a, b)));
  }

  // DotVecMat(Vec4r, VMatrix4x4r). 4x4 SIMD version.
  {
    Vec4r a(1_r, 2_r, 3_r, 4_r);
    VMatrix4x4r b = {
        Vec4r(10_r, 20_r, 30_r, 40_r),
        Vec4r(50_r, 60_r, 70_r, 80_r),
        Vec4r(90_r, 100_r, 110_r, 120_r),
        Vec4r(130_r, 140_r, 150_r, 160_r)};
    EXPECT_NEAR_EQ(Vec4r(900_r, 1000_r, 1100_r, 1200_r), DotVecMat4x4(a, b));
  }
}

TEST(MatrixUtils, Norm) {
  auto runTest = [](auto const& mat) {
    real normSqr = 0_r;
    for (int i = 0; i < mat.dims[0]; ++i) {
      for (int j = 0; j < mat.dims[1]; ++j) {
        normSqr += Sqr(mat[i][j]);
      }
    }

    real const norm = Sqrt(normSqr);
    EXPECT_NEAR_EQ(Norm(mat), norm);
    EXPECT_NEAR_EQ(Sqrt(NormSqr(mat)), norm);
  };

  // 2x2
  runTest(Matrix2x2r{Real2{1_r, -2_r}, Real2{-3_r, 4_r}});

  // 3x3
  runTest(
      Matrix3x3r{
          Real3{1_r, -2_r, 3_r}, //
          Real3{-4_r, 5_r, -6_r}, //
          Real3{7_r, -8_r, 9_r}});
}

TEST(MatrixUtils, Norm3x3) {
  auto expectedNormSqr3x3 = [](auto const& mat) {
    real normSqr = 0_r;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        normSqr += Sqr(mat[i][j]);
      }
    }
    return normSqr;
  };

  // 3x3 (actually 3x4) SIMD matrix
  {
    VMatrix3x3r m{
        Vec4r{1_r, -2_r, 3_r, -4_r}, Vec4r{5_r, -6_r, 7_r, -8_r}, Vec4r{9_r, -10_r, 11_r, -12_r}};
    EXPECT_NEAR_EQ(expectedNormSqr3x3(m), NormSqr3x3(m));
    EXPECT_NEAR_EQ(Sqrt(expectedNormSqr3x3(m)), Norm3x3(m));
  }

  // 3x3 portion of a 4x4 SIMD matrix
  {
    VMatrix4x4r m{
        Vec4r{1_r, -2_r, 3_r, -4_r},
        Vec4r{5_r, -6_r, 7_r, -8_r},
        Vec4r{9_r, -10_r, 11_r, -12_r},
        Vec4r{13_r, -14_r, 15_r, -16_r}};
    EXPECT_NEAR_EQ(expectedNormSqr3x3(m), NormSqr3x3(m));
    EXPECT_NEAR_EQ(Sqrt(expectedNormSqr3x3(m)), Norm3x3(m));
  }
}

TEST(MatrixUtils, DotMatVec) {
  // DotMatVec(Matrix2x3r, Real3)
  {
    constexpr Matrix2x3r a = {Real3{1_r, 2_r, 3_r}, Real3{4_r, 5_r, 6_r}};
    constexpr Real3 b = {7_r, 8_r, 9_r};
    static_assert(NearEqual(Real2{50_r, 122_r}, DotMatVec(a, b)));
  }

  // DotMatVec(Matrix3x3r, Real3)
  {
    constexpr Matrix3x3r a = {Real3{1_r, 2_r, 3_r}, Real3{4_r, 5_r, 6_r}, Real3{7_r, 8_r, 9_r}};
    constexpr Real3 b = {10_r, 11_r, 12_r};
    static_assert(NearEqual(Real3{68_r, 167_r, 266_r}, DotMatVec(a, b)));
  }

  // DotMatVec3x3(VMatrix3x3r, Vec4r). Only uses the first 3 vector components.
  {
    VMatrix3x3r a = {
        Vec4r(1_r, 2_r, 3_r, 111_r), Vec4r(4_r, 5_r, 6_r, 222_r), Vec4r(7_r, 8_r, 9_r, 333_r)};
    Vec4r b = Vec4r(10_r, 11_r, 12_r, 444_r);
    EXPECT_NEAR_EQ(Vec4r(68_r, 167_r, 266_r), ToSimdDirection(DotMatVec3x3(a, b)));
  }

  // DotMatVec3x3(VMatrix4x4r, Vec4r). Only uses upper-left 3x3 portion.
  {
    VMatrix4x4r a = {
        Vec4r(1_r, 2_r, 3_r, 111_r),
        Vec4r(4_r, 5_r, 6_r, 222_r),
        Vec4r(7_r, 8_r, 9_r, 333_r),
        Vec4r(10_r, 11_r, 12_r, 444_r)};
    Vec4r b = Vec4r(10_r, 11_r, 12_r, 444_r);
    EXPECT_NEAR_EQ(Vec4r(68_r, 167_r, 266_r), ToSimdDirection(DotMatVec3x3(a, b)));
  }

  // DotMatVec3xN(NdArray<Vec4r, D0> const& m, Vec4r v), with N = 4, D0 = 3, 4. (N = 3 cases tested
  // through DotMatVec3x3)
  {
    NdArray<Vec4r, 3> a3 = {
        Vec4r(1_r, 2_r, 3_r, 4_r), Vec4r(5_r, 6_r, 7_r, 8_r), Vec4r(9_r, 10_r, 11_r, 12_r)};
    NdArray<Vec4r, 4> a4 = {
        Vec4r(1_r, 2_r, 3_r, 4_r),
        Vec4r(5_r, 6_r, 7_r, 8_r),
        Vec4r(9_r, 10_r, 11_r, 12_r),
        Vec4r(111_r, 222_r, 333_r, 444_r)}; // Padding should be ignored.
    Vec4r b = Vec4r(13_r, 14_r, 15_r, 16_r);
    Vec4r ab(150_r, 382_r, 614_r);
    EXPECT_NEAR_EQ(ab, ToSimdDirection(DotMatVec3xN<4>(a3, b)));
    EXPECT_NEAR_EQ(ab, ToSimdDirection(DotMatVec3xN<4>(a4, b)));
  }

  // DotMatVec(Matrix4x4r, Real4)
  {
    constexpr Matrix4x4r a = {
        Real4{1.0_r, 2.0_r, 3.0_r, 4.0_r},
        Real4{1.1_r, 2.1_r, 3.1_r, 4.1_r},
        Real4{1.2_r, 2.2_r, 3.2_r, 4.2_r},
        Real4{1.3_r, 2.3_r, 3.3_r, 4.3_r}};
    constexpr Real4 b = {100_r, 200_r, 300_r, 400_r};
    static_assert(NearEqual(Real4{3000_r, 3100_r, 3200_r, 3300_r}, DotMatVec(a, b)));
  }

  // DotMatVec(VMatrix4x4r, Vec4r)
  {
    NdArray<Vec4r, 4> a = {
        Vec4r(1.0_r, 2.0_r, 3.0_r, 4.0_r),
        Vec4r(1.1_r, 2.1_r, 3.1_r, 4.1_r),
        Vec4r(1.2_r, 2.2_r, 3.2_r, 4.2_r),
        Vec4r(1.3_r, 2.3_r, 3.3_r, 4.3_r)};
    auto b = Vec4r(100_r, 200_r, 300_r, 400_r);
    auto answer = DotMatVec4x4(a, b);
    EXPECT_NEAR_EQ(Vec4r(3000_r, 3100_r, 3200_r, 3300_r), answer);
  }
}

// Test products of matrix MxN with vector N, with various implementations that use dynamic-size
// arrays.
TEST(MatrixUtils, DotMatVecDynamic) {
  constexpr int M = 5;
  constexpr int N = 6;

  NdArray<real, M, N> Astatic;
  for (int i = 0, v = 0; i < M; i++) {
    for (int j = 0; j < N; j++, v++) {
      Astatic[i][j] = (real)v;
    }
  }
  NdArray<real, N> bstatic;
  std::vector<real> bdynamic(N);
  for (int i = 0; i < N; i++) {
    bstatic[i] = (real)i;
    bdynamic[i] = (real)i;
  }
  Span<real const> bdynamicspan = MakeConstSpan(bdynamic);

  NdArray<real, M> res1 = DotMatVec(Astatic, bstatic);

  // NdArray_1d = DotMatVec(NdArray_2d, span)
  {
    NdArray<real, M> res2 = DotMatVec(Astatic, bdynamicspan);
    for (int i = 0; i < M; i++) {
      EXPECT_NEAR_EQ(res1[i], res2[i]);
    }
  }

  // DotMatVec(NdArray_2d, span, span& out)
  {
    std::vector<real> res2(M);
    Span<real> res2span = MakeSpan(res2);
    DotMatVec(Astatic, bdynamicspan, res2span);
    for (int i = 0; i < M; i++) {
      EXPECT_NEAR_EQ(res1[i], res2[i]);
    }
  }
}

TEST(MatrixUtils, Invert) {
  // 2x2
  constexpr Matrix2x2r identity2 = Eye<2>();
  constexpr Matrix2x2r m2 = {Real2{4_r, 7_r}, Real2{2_r, 6_r}};
  // Det(m2) = 24 - 14 = 10. Inverse is (1/10) * [[6, -7], [-2, 4]].
  constexpr Matrix2x2r m2Inv = {Real2{0.6_r, -0.7_r}, Real2{-0.2_r, 0.4_r}};
  static_assert(NearEqual(identity2, Invert(identity2, 1_r)));
  static_assert(NearEqual(identity2 / 2_r, Invert(identity2 * 2_r, 4_r)));
  static_assert(NearEqual(m2Inv, Invert(m2, Det(m2))));
  static_assert(NearEqual(m2Inv, Invert(m2)));
  static_assert(NearEqual(m2, Invert(m2Inv)));

  // 3x3
  constexpr Matrix3x3r identity3 = Eye<3>();
  constexpr Matrix3x3r m = {Real3{3_r, 1_r, 4_r}, Real3{1_r, 5_r, 9_r}, Real3{2_r, 6_r, 5_r}};
  constexpr Matrix3x3r m_inv = {
      Real3{0.3222222_r, -0.21111111_r, 0.12222222_r},
      Real3{-0.14444444_r, -0.07777778_r, 0.25555556_r},
      Real3{0.04444444_r, 0.17777778_r, -0.15555556_r}};
  static_assert(NearEqual(identity3, Invert(identity3, 1_r)));
  static_assert(NearEqual(identity3 / 2_r, Invert(identity3 * 2_r, 8_r)));
  static_assert(NearEqual(m_inv, Invert(m, Det(m))));
  static_assert(NearEqual(m, Invert(m_inv, Det(m_inv))));
}

static NdArray<real, 2, 2> MatrixToNdArray2x2(Matrix<real, 2, 2> const& m) {
  return NdArray<real, 2, 2>{Real2{m(0, 0), m(0, 1)}, Real2{m(1, 0), m(1, 1)}};
};

TEST(MatrixUtils, Det2x2) {
  // Relies on correctness of non-SIMD implementations of determinant for the Matrix
  // class, which is tested separately.
  Matrix<real, 2, 2> a;
  a(0, 0) = 1_r;
  a(0, 1) = 2_r;
  a(1, 0) = 3_r;
  a(1, 1) = 4_r;
  VMatrix2x2r const va = ToSimdMatrix(MatrixToNdArray2x2(a));
  real const det = Determinant(a);
  real const vdet = Det2x2(va);
  EXPECT_NEAR_EQ(det, vdet);
}

TEST(MatrixUtils, Invert2x2) {
  // Relies on correctness of non-SIMD implementations of inverse for the Matrix
  // class, which is tested separately.
  Matrix<real, 2, 2> a;
  a(0, 0) = 5_r;
  a(0, 1) = 6_r;
  a(1, 0) = 7_r;
  a(1, 1) = 8_r;
  VMatrix2x2r const va = ToSimdMatrix(MatrixToNdArray2x2(a));
  real const vdet = Det2x2(va);
  VMatrix2x2r vaInv = Invert2x2(va, vdet);
  EXPECT_NEAR_EQ(MatrixToNdArray2x2(Inverse(a)), ToNdArray2x2(vaInv));
  // Also test that product of a with aInv is identity.
  EXPECT_NEAR_EQ(ToNdArray2x2(Dot2x2(va, vaInv)), Eye<2>());
  EXPECT_NEAR_EQ(ToNdArray2x2(Dot2x2(vaInv, va)), Eye<2>());
}

TEST(MatrixUtils, InvertTransformation) {
  constexpr Matrix3x3r kAxes = Eye<3>();
  constexpr Real3 kScales[] = {
      Real3{2_r, 1_r, 1_r},
      Real3{1_r, 2_r, 1_r},
      Real3{1_r, 1_r, 2_r},
      Real3{0.2_r, -0.4_r, 0.6_r}, // mirrored about Y
      Real3{-0.2_r, 0.4_r, -0.6_r}}; // mirrored about X & Z
  constexpr Real3 kRotations[] = {
      Real3{kPI / 2_r, 0_r, 0_r},
      Real3{0_r, kPI / 2_r, 0_r},
      Real3{0_r, 0_r, kPI / 2_r},
      Real3{1_r, -2_r, 3_r}};
  constexpr Real3 kTranslations[] = {
      Real3{0_r, 0_r, 0_r},
      Real3{0.1_r, 0_r, 0_r},
      Real3{0_r, 0.1_r, 0_r},
      Real3{0_r, 0_r, 0.1_r},
      Real3{-0.1_r, 0.2_r, -0.3_r}};

  for (auto s : kScales) {
    for (auto r : kRotations) {
      for (auto t : kTranslations) {
        // The cases with non-90-degree rotations need greater tolerance.
        Vec4r tol = 3_r * kDefaultNearEqualEpsilon<real>;

        // Use TransformRT and scale to build a 4x4 matrix
        auto q = Quaternion::FromAxisAngle(kAxes[2], r[2]) *
            Quaternion::FromAxisAngle(kAxes[1], r[1]) * Quaternion::FromAxisAngle(kAxes[0], r[0]);
        auto rt = TransformRT{q, t};
        auto scaleMat = VDiagonalMatrix<4>(ToSimd(s, 1_r));
        auto mat = Dot4x4(ToVMatrix4x4(rt), scaleMat);

        // Compare InvertTransformation vs InvertTransformationTransposed
        auto matInv = InvertTransformation(mat);
        auto matInv2 = Transpose4x4(InvertTransformationTransposed(Transpose4x4(mat)));
        EXPECT_NEAR_TOL(matInv, matInv2, tol);

        // Compare InvertTransform vs Invert(VMatrix4x4r)
        EXPECT_NEAR_TOL(Invert4x4(mat), matInv, tol);

        // Compare InvertTransform vs Invert(TransformRT)
        auto scaleMatInv = VDiagonalMatrix<4>(1_r / ToSimd(s, 1_r));
        auto matInvFromRt = Dot4x4(scaleMatInv, ToVMatrix4x4(Invert(rt)));
        EXPECT_NEAR_TOL(matInvFromRt, matInv, tol);

        // mat * matInv --> identity
        EXPECT_NEAR_TOL(VEye<4>(), Dot4x4(mat, matInv), tol);
        EXPECT_NEAR_TOL(VEye<4>(), Dot4x4(matInv, mat), tol);
      }
    }
  }
}

TEST(MatrixUtils, Det) {
  // 2x2
  constexpr Matrix2x2r identity2 = {Real2{1_r, 0_r}, Real2{0_r, 1_r}};
  constexpr Matrix2x2r m2 = {Real2{1_r, 2_r}, Real2{3_r, 4_r}};
  static_assert(NearEqual(1_r, Det(identity2)));
  static_assert(NearEqual(4_r, Det(identity2 * 2_r)));
  static_assert(NearEqual(-2_r, Det(m2)));

  // 3x3
  constexpr Matrix3x3r identity3 = {
      Real3{1_r, 0_r, 0_r}, Real3{0_r, 1_r, 0_r}, Real3{0_r, 0_r, 1_r}};
  constexpr Matrix3x3r m = {Real3{3_r, 1_r, 4_r}, Real3{1_r, 5_r, 9_r}, Real3{2_r, 6_r, 5_r}};
  static_assert(NearEqual(1_r, Det(identity3)));
  static_assert(NearEqual(8_r, Det(identity3 * 2_r)));
  static_assert(NearEqual(-90_r, Det(m)));
}

TEST(MatrixUtils, Det3x3) {
  NdArray<Vec4r, 3> const m1 = {
      Vec4r(1_r, 2_r, 3_r, 4_r),
      Vec4r(5_r, 6_r, 7_r, 8_r),
      Vec4r(9_r, 10_r, 11_r, 12_r),
  };

  NdArray<Vec4r, 3> const m2 = {
      Vec4r(1_r, 0_r, 0_r, 0_r),
      Vec4r(0_r, 1_r, 0_r, 0_r),
      Vec4r(0_r, 0_r, 1_r, 0_r),
  };

  EXPECT_NEAR_EQ(0_r, Det3x3(m1));
  EXPECT_NEAR_EQ(1_r, Det3x3(m2));
}

TEST(MatrixUtils, Cofactor) {
  // 2x2 case
  {
    constexpr Matrix2x2r a = {Real2{1_r, 2_r}, Real2{3_r, 4_r}};
    static_assert(Cofactor(a) == Matrix2x2r{Real2{4_r, -3_r}, Real2{-2_r, 1_r}});

    // Cross-check against the defining identity A * Cofactor(A)^T = det(A) * I.
    constexpr real detA = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    static_assert(NearEqual(detA * Eye<2>(), Dot(a, Transpose(Cofactor(a)))));
  }

  // 3x3 case
  {
    constexpr Matrix3x3r a = {Real3{1_r, 2_r, 3_r}, Real3{0_r, 1_r, 4_r}, Real3{5_r, 6_r, 0_r}};
    constexpr Matrix3x3r expected = {
        Real3{-24_r, 20_r, -5_r}, Real3{18_r, -15_r, 4_r}, Real3{5_r, -4_r, 1_r}};
    static_assert(NearEqual(expected, Cofactor(a)));

    // Cross-check against the defining identity A * Cofactor(A)^T = det(A) * I.
    static_assert(NearEqual(Det(a) * Eye<3>(), Dot(a, Transpose(Cofactor(a)))));
  }
}

TEST(MatrixUtils, Outer) {
  // outer(Real2, Real2)
  {
    constexpr Real2 a = {2_r, 3_r};
    constexpr Real2 b = {4_r, 5_r};
    constexpr Matrix2x2r expected = {Real2{8_r, 10_r}, Real2{12_r, 15_r}};
    static_assert(NearEqual(expected, Outer(a, b)));
  }

  // outer(Real3, Real2)
  {
    constexpr Real3 a = {2_r, 3_r, 4_r};
    constexpr Real2 b = {5_r, 6_r};
    constexpr Matrix3x2r expected = {Real2{10_r, 12_r}, Real2{15_r, 18_r}, Real2{20_r, 24_r}};
    static_assert(NearEqual(expected, Outer(a, b)));
  }

  // outer(Real3, Real3)
  {
    constexpr Real3 a = {2_r, 3_r, 4_r};
    constexpr Real3 b = {5_r, 6_r, 7_r};
    constexpr Matrix3x3r expected = {
        Real3{10_r, 12_r, 14_r}, Real3{15_r, 18_r, 21_r}, Real3{20_r, 24_r, 28_r}};
    static_assert(NearEqual(expected, Outer(a, b)));
  }
}

TEST(MatrixUtils, Outer3) {
  Vec4r a(2_r, 3_r, 4_r, 5_r);
  Vec4r b(6_r, 7_r, 8_r, 9_r);

  // VOuter3 considres a & b to be 3-component vectors, so it creates
  // 3 rows of output. It technically still creates the same 4th column
  // of output because it would be more work to do otherwise, but the
  // caller is expected to ignore those values.
  auto outer3 = Outer3(a, b);
  EXPECT_EQ(3, outer3.size());
  EXPECT_NEAR_EQ(Vec4r(12_r, 14_r, 16_r, 18_r), outer3[0]);
  EXPECT_NEAR_EQ(Vec4r(18_r, 21_r, 24_r, 27_r), outer3[1]);
  EXPECT_NEAR_EQ(Vec4r(24_r, 28_r, 32_r, 36_r), outer3[2]);
}

TEST(MatrixUtils, Colon) {
  // 2x2
  constexpr Matrix2x2r C = {Real2{1_r, 2_r}, Real2{3_r, 4_r}};
  static_assert(NearEqual(30_r, Colon(C, C)));

  // 3x3
  constexpr Matrix3x3r A = {Real3{1_r, 2_r, 3_r}, Real3{4_r, 5_r, 6_r}, Real3{7_r, 8_r, 9_r}};
  constexpr Matrix3x3r B = {Real3{9_r, 8_r, 7_r}, Real3{6_r, 5_r, 4_r}, Real3{3_r, 2_r, 1_r}};
  // A:B = 1*9 + 2*8 + 3*7 + 4*6 + 5*5 + 6*4 + 7*3 + 8*2 + 9*1 = 165
  static_assert(NearEqual(165_r, Colon(A, B)));
  // A:A = 1+4+9+16+25+36+49+64+81 = 285
  static_assert(NearEqual(285_r, Colon(A, A)));

  // Frobenius identity: A:I = trace(A)
  static_assert(NearEqual(Trace(A), Colon(A, Eye<3>())));
}

TEST(MatrixUtils, Colon3x3) {
  Vec4r v = {1_r, 2_r, 3_r, 4_r};
  VMatrix3x3r A = {v, 2_r * v, 3_r * v};
  VMatrix3x3r B = 4_r * A;
  EXPECT_NEAR_EQ(784_r, Colon3x3(A, B));
}

TEST(MatrixUtils, SymMatrix2x2) {
  constexpr Matrix2x2r expected = {Real2{1_r, 2_r}, Real2{2_r, 3_r}};
  static_assert(NearEqual(expected, SymMatrix2x2(1_r, 2_r, 3_r)));

  // Symmetry property: M[0][1] == M[1][0]
  constexpr Matrix2x2r m = SymMatrix2x2(7_r, -3_r, 4_r);
  static_assert(NearEqual(m[0][1], m[1][0]));
}

TEST(MatrixUtils, Skew) {
  //  [  0  -v2   v1 ]
  //  [  v2   0  -v0 ]
  //  [ -v1   v0   0 ]
  constexpr Real3 v = {1_r, 2_r, 3_r};
  static_assert(
      Skew(v) == Matrix3x3r{Real3{0_r, -3_r, 2_r}, Real3{3_r, 0_r, -1_r}, Real3{-2_r, 1_r, 0_r}});
}

TEST(MatrixUtils, Trace) {
  constexpr Matrix3x3r m = {Real3{1_r, 2_r, 3_r}, Real3{4_r, 5_r, 6_r}, Real3{7_r, 8_r, 9_r}};
  static_assert(NearEqual(15_r, Trace(m)));
  static_assert(NearEqual(30_r, Trace(m * 2_r)));
}

TEST(MatrixUtils, Trace3x3) {
  constexpr Matrix3x3r m = {Real3{10_r, 1_r, 2_r}, Real3{3_r, 20_r, 4_r}, Real3{5_r, 6_r, 30_r}};
  auto const vm = ToSimdMatrix(m);
  EXPECT_NEAR_EQ(60_r, Trace3x3(vm));
}

TEST(MatrixUtils, Trace2x2) {
  Matrix2x2r m = {Real2{1_r, 222_r}, Real2{333_r, 4_r}};
  EXPECT_NEAR_EQ(5_r, Trace2x2(ToSimdMatrix(m)));
  EXPECT_NEAR_EQ(10_r, Trace2x2(ToSimdMatrix(m * 2_r)));
}

TEST(MatrixUtils, PseudoInvert) {
  // Compare values from pyMochi
  Matrix3x2r const a = {Real2{1_r, 2_r}, Real2{3_r, 4_r}, Real2{5_r, 6_r}};
  Matrix2x3r inv;
  real det = 0_r;
  PseudoInvert(a, &inv, &det);
  EXPECT_TRUE(NearEqual(
      Matrix2x3r{
          Real3{-1.33333333_r, -0.33333333_r, 0.66666667_r},
          Real3{1.08333333_r, 0.33333333_r, -0.41666667_r}},
      inv));
  EXPECT_TRUE(NearEqual(4.89897949_r, det));
}

TEST(MatrixUtils, DiagonalMatrix) {
  static_assert(NearEqual(NdArray<real, 1, 1>{Real1{1.23_r}}, DiagonalMatrix<1>(1.23_r)));
  static_assert(NearEqual(
      NdArray<real, 2, 2>{Real2{2.34_r, 0_r}, Real2{0_r, 2.34_r}}, DiagonalMatrix<2>(2.34_r)));
  static_assert(NearEqual(
      NdArray<real, 3, 3>{
          Real3{3.45_r, 0_r, 0_r}, Real3{0_r, 3.45_r, 0_r}, Real3{0_r, 0_r, 3.45_r}},
      DiagonalMatrix<3>(3.45_r)));
  static_assert(NearEqual(
      NdArray<real, 4, 4>{
          Real4{8_r, 0_r, 0_r, 0_r},
          Real4{0_r, 8_r, 0_r, 0_r},
          Real4{0_r, 0_r, 8_r, 0_r},
          Real4{0_r, 0_r, 0_r, 8_r}},
      DiagonalMatrix<4>(8_r)));
}

TEST(MatrixUtils, DiagonalMatrixArray) {
  static_assert(NearEqual(
      NdArray<real, 2, 2>{Real2{2.34_r, 0_r}, Real2{0_r, 3.45_r}},
      DiagonalMatrix(Real2{2.34_r, 3.45_r})));
  static_assert(NearEqual(
      NdArray<real, 3, 3>{
          Real3{3.45_r, 0_r, 0_r}, Real3{0_r, 4.56_r, 0_r}, Real3{0_r, 0_r, 5.67_r}},
      DiagonalMatrix(Real3{3.45_r, 4.56_r, 5.67_r})));
  static_assert(NearEqual(
      NdArray<real, 4, 4>{
          Real4{8_r, 0_r, 0_r, 0_r},
          Real4{0_r, 9_r, 0_r, 0_r},
          Real4{0_r, 0_r, 10_r, 0_r},
          Real4{0_r, 0_r, 0_r, 11_r}},
      DiagonalMatrix(Real4{8_r, 9_r, 10_r, 11_r})));
}

TEST(MatrixUtils, Eye) {
  static_assert(NearEqual(NdArray<real, 1, 1>{Real1{1_r}}, Eye<1>()));
  static_assert(NearEqual(NdArray<real, 2, 2>{Real2{1_r, 0_r}, Real2{0_r, 1_r}}, Eye<2>()));
  static_assert(NearEqual(
      NdArray<real, 3, 3>{Real3{1_r, 0_r, 0_r}, Real3{0_r, 1_r, 0_r}, Real3{0_r, 0_r, 1_r}},
      Eye<3>()));
  static_assert(NearEqual(
      NdArray<real, 4, 4>{
          Real4{1_r, 0_r, 0_r, 0_r},
          Real4{0_r, 1_r, 0_r, 0_r},
          Real4{0_r, 0_r, 1_r, 0_r},
          Real4{0_r, 0_r, 0_r, 1_r}},
      Eye<4>()));
}

static constexpr Matrix3x3r testMatrices[] = {
    // Identity
    Matrix3x3r{
        Real3{1.000000e+00_r, 0.000000e+00_r, 0.000000e+00_r},
        Real3{0.000000e+00_r, 1.000000e+00_r, 0.000000e+00_r},
        Real3{0.000000e+00_r, 0.000000e+00_r, 1.000000e+00_r}},
    // Purely random
    Matrix3x3r{
        Real3{1.757904e+00_r, 8.375675e-01_r, 8.940403e-01_r},
        Real3{7.781580e-01_r, 1.293040e+00_r, 8.082672e-01_r},
        Real3{9.259674e-01_r, 3.419398e-01_r, 1.178375e+00_r}},
    // Isochoric
    Matrix3x3r{
        Real3{1.852487e+00_r, 3.954069e-01_r, 8.651419e-01_r},
        Real3{1.217431e-01_r, 1.074060e+00_r, 2.034998e-01_r},
        Real3{1.669658e-01_r, 6.375304e-01_r, 1.053620e+00_r}},
    // Symmetric
    Matrix3x3r{
        Real3{3.474406e+00_r, 9.696913e-01_r, 1.803357e+00_r},
        Real3{9.696913e-01_r, 1.716397e+00_r, 1.232369e+00_r},
        Real3{1.803357e+00_r, 1.232369e+00_r, 1.899997e+00_r}},
    // Pure dilation
    Matrix3x3r{
        Real3{4.000000e+00_r, 0.000000e+00_r, 0.000000e+00_r},
        Real3{0.000000e+00_r, 4.000000e+00_r, 0.000000e+00_r},
        Real3{0.000000e+00_r, 0.000000e+00_r, 4.000000e+00_r}},
    // Pure contraction
    Matrix3x3r{
        Real3{2.500000e-01_r, 0.000000e+00_r, 0.000000e+00_r},
        Real3{0.000000e+00_r, 2.500000e-01_r, 0.000000e+00_r},
        Real3{0.000000e+00_r, 0.000000e+00_r, 2.500000e-01_r}},
    // Shear
    Matrix3x3r{
        Real3{1.000000e+00_r, 5.000000e-01_r, 0.000000e+00_r},
        Real3{5.000000e-01_r, 1.000000e+00_r, 0.000000e+00_r},
        Real3{0.000000e+00_r, 0.000000e+00_r, 1.000000e+00_r}}};

TEST(MathUtils, Dot3x3) {
  // This test relies on the correctness of the scalar implementation (tested elsewhere in this
  // file). It simply compares the SIMD and non-SIMD output.
  for (auto A : testMatrices) {
    for (auto B : testMatrices) {
      auto expected = Dot(A, B);
      auto actual = ToNdArray3x3(Dot3x3(ToSimdMatrix(A), ToSimdMatrix(B)));
      EXPECT_NEAR_EQ(expected, actual);
    }
  }
}

static constexpr Matrix2x2r testMatrices2x2[] = {
    Matrix2x2r{Real2{1.000000e+00_r, 0.000000e+00_r}, Real2{0.000000e+00_r, 1.000000e+00_r}},
    Matrix2x2r{Real2{1.000000e+00_r, 2.000000e+00_r}, Real2{3.000000e+00_r, 4.000000e+00_r}},
    Matrix2x2r{Real2{5.000000e+00_r, 6.000000e+00_r}, Real2{7.000000e+00_r, 8.000000e+00_r}}};

TEST(MathUtils, Dot2x2) {
  // This test relies on the correctness of the scalar implementation (tested elsewhere in this
  // file). It simply compares the SIMD and non-SIMD output.
  for (auto A : testMatrices2x2) {
    for (auto B : testMatrices2x2) {
      auto expected = Dot(A, B);
      auto actual = ToNdArray2x2(Dot2x2(ToSimdMatrix(A), ToSimdMatrix(B)));
      EXPECT_NEAR_EQ(expected, actual);
    }
  }
}

TEST(MathUtils, Outer2) {
  Matrix2x2r const a = {Real2{1_r, 2_r}, Real2{3_r, 4_r}};
  Matrix2x2r const b = {Real2{5_r, 6_r}, Real2{7_r, 8_r}};
  auto actual = Outer2(ToSimdMatrix(a), ToSimdMatrix(b));
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 2; k++) {
        for (int l = 0; l < 2; l++) {
          EXPECT_NEAR_EQ(a[i][j] * b[k][l], actual[i][j][2 * k + l]);
        } // l
      } // k
    } // j
  } // i
}

TEST(MatrixUtils, Outer1st2nd) {
  constexpr Real3 v{1_r, 2_r, 3_r};
  constexpr Matrix3x3r M{Real3{1_r, 2_r, 3_r}, Real3{4_r, 5_r, 6_r}, Real3{7_r, 8_r, 9_r}};
  constexpr NdArray<real, 3, 3, 3> P{
      Matrix3x3r{Real3{1_r, 2_r, 3_r}, Real3{4_r, 5_r, 6_r}, Real3{7_r, 8_r, 9_r}},
      Matrix3x3r{Real3{2_r, 4_r, 6_r}, Real3{8_r, 10_r, 12_r}, Real3{14_r, 16_r, 18_r}},
      Matrix3x3r{Real3{3_r, 6_r, 9_r}, Real3{12_r, 15_r, 18_r}, Real3{21_r, 24_r, 27_r}}};
  VTensor3x3x3r PTestSIMD = Outer3(ToSimd(v), ToSimdMatrix(M));
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(Norm3x3(ToSimdMatrix(P[i]) - PTestSIMD[i]), 0_r, 1e-6f);
  }
}

TEST(MatrixUtils, Outer2nd2nd) {
  constexpr Matrix3x3r M0{Real3{1_r, 1_r, 1_r}, Real3{2_r, 2_r, 2_r}, Real3{3_r, 3_r, 3_r}};
  constexpr Matrix3x3r M1{Real3{1_r, 2_r, 3_r}, Real3{4_r, 5_r, 6_r}, Real3{7_r, 8_r, 9_r}};
  constexpr NdArray<real, 3, 3, 3, 3> P01{
      NdArray<real, 3, 3, 3>{
          Matrix3x3r{Real3{1_r, 1_r, 1_r}, Real3{2_r, 2_r, 2_r}, Real3{3_r, 3_r, 3_r}},
          Matrix3x3r{Real3{2_r, 2_r, 2_r}, Real3{4_r, 4_r, 4_r}, Real3{6_r, 6_r, 6_r}},
          Matrix3x3r{Real3{3_r, 3_r, 3_r}, Real3{6_r, 6_r, 6_r}, Real3{9_r, 9_r, 9_r}}},
      NdArray<real, 3, 3, 3>{
          Matrix3x3r{Real3{4_r, 4_r, 4_r}, Real3{8_r, 8_r, 8_r}, Real3{12_r, 12_r, 12_r}},
          Matrix3x3r{Real3{5_r, 5_r, 5_r}, Real3{10_r, 10_r, 10_r}, Real3{15_r, 15_r, 15_r}},
          Matrix3x3r{Real3{6_r, 6_r, 6_r}, Real3{12_r, 12_r, 12_r}, Real3{18_r, 18_r, 18_r}}},
      NdArray<real, 3, 3, 3>{
          Matrix3x3r{Real3{7_r, 7_r, 7_r}, Real3{14_r, 14_r, 14_r}, Real3{21_r, 21_r, 21_r}},
          Matrix3x3r{Real3{8_r, 8_r, 8_r}, Real3{16_r, 16_r, 16_r}, Real3{24_r, 24_r, 24_r}},
          Matrix3x3r{Real3{9_r, 9_r, 9_r}, Real3{18_r, 18_r, 18_r}, Real3{27_r, 27_r, 27_r}}}};
  constexpr NdArray<real, 3, 3, 3, 3> P10{
      NdArray<real, 3, 3, 3>{
          Matrix3x3r{Real3{1_r, 2_r, 3_r}, Real3{4_r, 5_r, 6_r}, Real3{7_r, 8_r, 9_r}},
          Matrix3x3r{Real3{1_r, 2_r, 3_r}, Real3{4_r, 5_r, 6_r}, Real3{7_r, 8_r, 9_r}},
          Matrix3x3r{Real3{1_r, 2_r, 3_r}, Real3{4_r, 5_r, 6_r}, Real3{7_r, 8_r, 9_r}}},
      NdArray<real, 3, 3, 3>{
          Matrix3x3r{Real3{2_r, 4_r, 6_r}, Real3{8_r, 10_r, 12_r}, Real3{14_r, 16_r, 18_r}},
          Matrix3x3r{Real3{2_r, 4_r, 6_r}, Real3{8_r, 10_r, 12_r}, Real3{14_r, 16_r, 18_r}},
          Matrix3x3r{Real3{2_r, 4_r, 6_r}, Real3{8_r, 10_r, 12_r}, Real3{14_r, 16_r, 18_r}}},
      NdArray<real, 3, 3, 3>{
          Matrix3x3r{Real3{3_r, 6_r, 9_r}, Real3{12_r, 15_r, 18_r}, Real3{21_r, 24_r, 27_r}},
          Matrix3x3r{Real3{3_r, 6_r, 9_r}, Real3{12_r, 15_r, 18_r}, Real3{21_r, 24_r, 27_r}},
          Matrix3x3r{Real3{3_r, 6_r, 9_r}, Real3{12_r, 15_r, 18_r}, Real3{21_r, 24_r, 27_r}}}};
  VTensor3x3x3x3r P01TestSIMD = Outer3(ToSimdMatrix(M0), ToSimdMatrix(M1));
  VTensor3x3x3x3r P10TestSIMD = Outer3(ToSimdMatrix(M1), ToSimdMatrix(M0));
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR(Norm3x3(ToSimdMatrix(P01[i][j]) - P01TestSIMD[i][j]), 0_r, 1e-6f);
      EXPECT_NEAR(Norm3x3(ToSimdMatrix(P10[i][j]) - P10TestSIMD[i][j]), 0_r, 1e-6f);
    }
  }
}

TEST(MatrixUtils, Outer1st3rd) {
  constexpr Real3 v{1_r, 2_r, 3_r};
  constexpr NdArray<real, 3, 3, 3> T{
      Matrix3x3r{Real3{1_r, 1_r, 1_r}, Real3{2_r, 2_r, 2_r}, Real3{3_r, 3_r, 3_r}},
      Matrix3x3r{Real3{2_r, 2_r, 2_r}, Real3{4_r, 4_r, 4_r}, Real3{6_r, 6_r, 6_r}},
      Matrix3x3r{Real3{3_r, 3_r, 3_r}, Real3{6_r, 6_r, 6_r}, Real3{9_r, 9_r, 9_r}}};
  constexpr NdArray<real, 3, 3, 3, 3> vT{
      NdArray<real, 3, 3, 3>{
          1_r * Matrix3x3r{Real3{1_r, 1_r, 1_r}, Real3{2_r, 2_r, 2_r}, Real3{3_r, 3_r, 3_r}},
          1_r * Matrix3x3r{Real3{2_r, 2_r, 2_r}, Real3{4_r, 4_r, 4_r}, Real3{6_r, 6_r, 6_r}},
          1_r * Matrix3x3r{Real3{3_r, 3_r, 3_r}, Real3{6_r, 6_r, 6_r}, Real3{9_r, 9_r, 9_r}}},
      NdArray<real, 3, 3, 3>{
          2_r * Matrix3x3r{Real3{1_r, 1_r, 1_r}, Real3{2_r, 2_r, 2_r}, Real3{3_r, 3_r, 3_r}},
          2_r * Matrix3x3r{Real3{2_r, 2_r, 2_r}, Real3{4_r, 4_r, 4_r}, Real3{6_r, 6_r, 6_r}},
          2_r * Matrix3x3r{Real3{3_r, 3_r, 3_r}, Real3{6_r, 6_r, 6_r}, Real3{9_r, 9_r, 9_r}}},
      NdArray<real, 3, 3, 3>{
          3_r * Matrix3x3r{Real3{1_r, 1_r, 1_r}, Real3{2_r, 2_r, 2_r}, Real3{3_r, 3_r, 3_r}},
          3_r * Matrix3x3r{Real3{2_r, 2_r, 2_r}, Real3{4_r, 4_r, 4_r}, Real3{6_r, 6_r, 6_r}},
          3_r * Matrix3x3r{Real3{3_r, 3_r, 3_r}, Real3{6_r, 6_r, 6_r}, Real3{9_r, 9_r, 9_r}}}};
  constexpr NdArray<real, 3, 3, 3, 3> Tv{
      NdArray<real, 3, 3, 3>{
          1_r * Matrix3x3r{Real3{1_r, 1_r, 1_r}, Real3{2_r, 2_r, 2_r}, Real3{3_r, 3_r, 3_r}},
          2_r * Matrix3x3r{Real3{1_r, 1_r, 1_r}, Real3{2_r, 2_r, 2_r}, Real3{3_r, 3_r, 3_r}},
          3_r * Matrix3x3r{Real3{1_r, 1_r, 1_r}, Real3{2_r, 2_r, 2_r}, Real3{3_r, 3_r, 3_r}}},
      NdArray<real, 3, 3, 3>{
          1_r * Matrix3x3r{Real3{2_r, 2_r, 2_r}, Real3{4_r, 4_r, 4_r}, Real3{6_r, 6_r, 6_r}},
          2_r * Matrix3x3r{Real3{2_r, 2_r, 2_r}, Real3{4_r, 4_r, 4_r}, Real3{6_r, 6_r, 6_r}},
          3_r * Matrix3x3r{Real3{2_r, 2_r, 2_r}, Real3{4_r, 4_r, 4_r}, Real3{6_r, 6_r, 6_r}}},
      NdArray<real, 3, 3, 3>{
          1_r * Matrix3x3r{Real3{3_r, 3_r, 3_r}, Real3{6_r, 6_r, 6_r}, Real3{9_r, 9_r, 9_r}},
          2_r * Matrix3x3r{Real3{3_r, 3_r, 3_r}, Real3{6_r, 6_r, 6_r}, Real3{9_r, 9_r, 9_r}},
          3_r * Matrix3x3r{Real3{3_r, 3_r, 3_r}, Real3{6_r, 6_r, 6_r}, Real3{9_r, 9_r, 9_r}}}};
  VTensor3x3x3r TSIMD = ToSimdTensor(T);
  Vec4r vSIMD = ToSimd(v);
  VTensor3x3x3x3r vTTestSIMD = Outer3(vSIMD, TSIMD);
  VTensor3x3x3x3r TvTestSIMD = Outer3(TSIMD, vSIMD);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR(Norm3x3(ToSimdMatrix(vT[i][j]) - vTTestSIMD[i][j]), 0_r, 1e-6f);
      EXPECT_NEAR(Norm3x3(ToSimdMatrix(Tv[i][j]) - TvTestSIMD[i][j]), 0_r, 1e-6f);
    }
  }
}

TEST(MatrixUtils, Outer3MatrixVector) {
  // Test matrix-vector Outer3 product: returns a 3x3x3 tensor
  // result[i][j][k] = mat[i][j] * vec[k]

  VMatrix3x3r mat = {
      Vec4r(1_r, 2_r, 3_r, 0_r), Vec4r(4_r, 5_r, 6_r, 0_r), Vec4r(7_r, 8_r, 9_r, 0_r)};

  Vec4r vec = Vec4r(2_r, 3_r, 5_r, 0_r);

  VTensor3x3x3r result = Outer3(mat, vec);

  // Verify the outer product manually
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        real expected = mat[i][j] * vec[k];
        EXPECT_NEAR_EQ(expected, result[i][j][k]);
      }
    }
  }
}

TEST(MatrixUtils, VDSkew3Antisymmetry) {
  VTensor3x3x3r dskew = VDSkew3();

  // Derivative of skew is the negation of the Levi-Civita symbol, which is uniquely defined by its
  // 012 component and full antisymmetry
  EXPECT_NEAR_EQ(dskew[0][1][2], -1_r);

  // Brute-force test of full antisymmetry:
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        // Verify antisymmetry in the last two indices
        real val_jk = dskew[i][j][k];
        real val_kj = dskew[i][k][j];
        EXPECT_NEAR_EQ(val_jk, -val_kj);
        // Verify antisymmetry in the first two indices
        real val_ij = dskew[i][j][k];
        real val_ji = dskew[j][i][k];
        EXPECT_NEAR_EQ(val_ij, -val_ji);
        // Verify antisymmetry in the first and last indices
        real val_ik = dskew[i][j][k];
        real val_ki = dskew[k][j][i];
        EXPECT_NEAR_EQ(val_ik, -val_ki);
      }
    }
  }

  // Additional verification: VDSkew3() should compute the derivatives of Skew3
  // For each basis vector e_i, Skew3(e_i) should equal dskew[i]
  Vec4r e0 = Vec4r(1_r, 0_r, 0_r, 0_r);
  Vec4r e1 = Vec4r(0_r, 1_r, 0_r, 0_r);
  Vec4r e2 = Vec4r(0_r, 0_r, 1_r, 0_r);

  VMatrix3x3r skew0 = Skew3(e0);
  VMatrix3x3r skew1 = Skew3(e1);
  VMatrix3x3r skew2 = Skew3(e2);

  EXPECT_NEAR_EQ(skew0, dskew[0]);
  EXPECT_NEAR_EQ(skew1, dskew[1]);
  EXPECT_NEAR_EQ(skew2, dskew[2]);
}

// Helper: verify DNormalize against centered finite differences.
template <size_t N, size_t kCount>
void TestDNormalizeFiniteDifference(NdArray<real, N> const (&testVectors)[kCount]) {
  real constexpr kEps = 1e-3_r;
  real constexpr kTol = 1e-3_r;

  for (auto const& v : testVectors) {
    auto const D = DNormalize(v);

    // Centered finite difference for each component.
    for (size_t i = 0; i < N; ++i) {
      auto vPlus = v;
      auto vMinus = v;
      vPlus[i] += kEps;
      vMinus[i] -= kEps;
      auto const fdDeriv = (Normalize(vPlus) - Normalize(vMinus)) / (2_r * kEps);
      for (size_t j = 0; j < N; ++j) {
        EXPECT_NEAR(fdDeriv[j], D[j][i], kTol);
      }
    }

    // Verify the precomputed-sqrNorm overload gives the same result.
    EXPECT_NEAR_EQ(D, DNormalize(v, NormSqr(v)));
  }
}

TEST(MatrixUtils, DNormalize) {
  {
    Real2 const vecs[] = {{1_r, 2_r}, {3_r, -1_r}, {-1_r, 2_r}, {5_r, 0_r}, {0_r, 7_r}};
    TestDNormalizeFiniteDifference(vecs);
  }
  {
    Real3 const vecs[] = {
        {1_r, 2_r, 3_r}, {3_r, 1_r, 2_r}, {-1_r, 2_r, -3_r}, {5_r, 0_r, 0_r}, {0_r, 0_r, 7_r}};
    TestDNormalizeFiniteDifference(vecs);
  }
  {
    Real4 const vecs[] = {
        {1_r, 2_r, 3_r, 4_r},
        {3_r, -1_r, 2_r, 1_r},
        {-1_r, 2_r, -3_r, 0.5_r},
        {5_r, 0_r, 0_r, 0_r}};
    TestDNormalizeFiniteDifference(vecs);
  }
}

TEST(MatrixUtils, DNormalize3) {
  // Test that DNormalize3 is consistent with finite difference approximation
  // of the derivative of Normalize<3>

  // Test with several different vectors
  DynamicArray<Vec4r> testVectors = {
      Vec4r(1_r, 2_r, 3_r, 0_r),
      Vec4r(3_r, 1_r, 2_r, 0_r),
      Vec4r(-1_r, 2_r, -3_r, 0_r),
      Vec4r(5_r, 0_r, 0_r, 0_r),
      Vec4r(0_r, 0_r, 7_r, 0_r)};

  real constexpr kEps = 1e-3_r;
  real constexpr kTol = 1e-3_r;

  for (auto const& v : testVectors) {
    // Skip near-zero vectors
    if (Norm<3>(v) < 1e-6_r) {
      continue;
    }

    VMatrix3x3r D = DNormalize3(v);

    // Compute finite difference approximation for each component
    for (int i = 0; i < 3; ++i) {
      // No non-const access to individual components of Vec4r
      Real3 vPlusR3 = ToReal3(v);
      Real3 vMinusR3 = ToReal3(v);
      vPlusR3[i] += kEps;
      vMinusR3[i] -= kEps;
      Vec4r vPlus = ToSimd(vPlusR3);
      Vec4r vMinus = ToSimd(vMinusR3);

      Vec4r vPlusNorm = Normalize<3>(vPlus);
      Vec4r vMinusNorm = Normalize<3>(vMinus);

      // Centered difference:
      Vec4r fdDeriv = (vPlusNorm - vMinusNorm) / (2_r * kEps);

      // Compare with analytical derivative
      for (int j = 0; j < 3; ++j) {
        EXPECT_NEAR(D[j][i], fdDeriv[j], kTol);
      }
    }
  }
}

TEST(MatrixUtils, LargestRow) {
  // 2x2 square
  {
    constexpr Matrix2x2r m = {Real2{1_r, 0_r}, Real2{3_r, 4_r}};
    real sqrNorm = 0_r;
    auto const row = LargestRow(m, &sqrNorm);
    EXPECT_NEAR_EQ(Real2(3_r, 4_r), row);
    EXPECT_NEAR_EQ(25_r, sqrNorm);
  }

  // 3x3 square
  {
    constexpr Matrix3x3r m = {Real3{1_r, 0_r, 0_r}, Real3{0_r, 5_r, 0_r}, Real3{2_r, 0_r, 2_r}};
    real sqrNorm = 0_r;
    auto const row = LargestRow(m, &sqrNorm);
    EXPECT_NEAR_EQ(Real3(0_r, 5_r, 0_r), row);
    EXPECT_NEAR_EQ(25_r, sqrNorm);
  }

  // 3x2 non-square (D0 > D1)
  {
    constexpr Matrix3x2r m = {Real2{1_r, 0_r}, Real2{0_r, 1_r}, Real2{3_r, 4_r}};
    real sqrNorm = 0_r;
    auto const row = LargestRow(m, &sqrNorm);
    EXPECT_NEAR_EQ(Real2(3_r, 4_r), row);
    EXPECT_NEAR_EQ(25_r, sqrNorm);
  }

  // 2x3 non-square (D0 < D1)
  {
    constexpr Matrix2x3r m = {Real3{3_r, 4_r, 0_r}, Real3{1_r, 0_r, 0_r}};
    real sqrNorm = 0_r;
    auto const row = LargestRow(m, &sqrNorm);
    EXPECT_NEAR_EQ(Real3(3_r, 4_r, 0_r), row);
    EXPECT_NEAR_EQ(25_r, sqrNorm);
  }

  // Null sqrNorm pointer
  {
    constexpr Matrix2x2r m = {Real2{1_r, 0_r}, Real2{3_r, 4_r}};
    auto const row = LargestRow(m, static_cast<real*>(nullptr));
    EXPECT_NEAR_EQ(Real2(3_r, 4_r), row);
  }
}
