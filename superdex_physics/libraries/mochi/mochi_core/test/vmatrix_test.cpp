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
#include <mochi_core/utils/nd_array_utils.h>
#include <mochi_core/utils/vmatrix.h>

using namespace mochi;

static_assert(alignof(VMatrix4x4r) == alignof(Vec4r), "Unexpected alignment");
static_assert(sizeof(VMatrix4x4r) == sizeof(real) * 16, "Unexpected padding");
static_assert(std::is_trivially_copyable_v<VMatrix4x4r>);

TEST(BasicTypes, LoadMatrix) {
  // NdArray<real, 2, 4>
  {
    // Load 8 reals into SIMD with no padding
    NdArray<real, 2, 4> dense = {Real4{1_r, 2_r, 3_r, 4_r}, Real4{5_r, 6_r, 7_r, 8_r}};
    NdArray<Vec4r, 2> simd1, simd2, simd3;
    LoadMatrix(simd1, dense);
    LoadMatrix<2, 4>(simd2, Flatten(dense)); // Dims must be explicit this time
    LoadMatrix<2, 4>(simd3, &dense[0][0]); // Dims must be explicit this time

    EXPECT_EQ(Vec4r(1_r, 2_r, 3_r, 4_r), simd1[0]);
    EXPECT_EQ(Vec4r(5_r, 6_r, 7_r, 8_r), simd1[1]);
    EXPECT_EQ(simd1, simd2);
    EXPECT_EQ(simd1, simd3);
  }

  // NdArray<real, 4, 3>
  {
    // Load 12 reals into SIMD with padding.
    NdArray<real, 4, 3> dense = {
        Real3{1_r, 2_r, 3_r}, Real3{4_r, 5_r, 6_r}, Real3{7_r, 8_r, 9_r}, Real3{10_r, 11_r, 12_r}};
    NdArray<Vec4r, 4> simd1, simd2, simd3;
    LoadMatrix(simd1, dense);
    LoadMatrix<4, 3>(simd2, Flatten(dense)); // Dims must be explicit this time
    LoadMatrix<4, 3>(simd3, &dense[0][0]); // Dims must be explicit this time

    // The 4th SIMD component normally has an unspecified value.
    // Set it to zero for comparison.
    for (int i = 0; i < 4; ++i) {
      simd1[i] &= SimdMask<Vec4r>(true, true, true, false);
      simd2[i] &= SimdMask<Vec4r>(true, true, true, false);
      simd3[i] &= SimdMask<Vec4r>(true, true, true, false);
    }

    EXPECT_EQ(Vec4r(1_r, 2_r, 3_r, 0_r), simd1[0]);
    EXPECT_EQ(Vec4r(4_r, 5_r, 6_r, 0_r), simd1[1]);
    EXPECT_EQ(Vec4r(7_r, 8_r, 9_r, 0_r), simd1[2]);
    EXPECT_EQ(Vec4r(10_r, 11_r, 12_r, 0_r), simd1[3]);
    EXPECT_EQ(simd1, simd2);
    EXPECT_EQ(simd1, simd3);
  }

  // NdArray<real, 2, 3, 4, 3>
  {
    NdArray<real, 2, 3, 4, 3> dense;
    real value = 1_r;
    for (int i = 0; i < dense.dims[0]; ++i) {
      for (int j = 0; j < dense.dims[1]; ++j) {
        for (int k = 0; k < dense.dims[2]; ++k) {
          for (int l = 0; l < dense.dims[3]; ++l) {
            dense[i][j][k][l] = value;
            value += 1_r;
          }
        }
      }
    }

    NdArray<Vec4r, 2, 3, 4> simd1, simd2;
    LoadMatrix(simd1, dense);
    LoadMatrix<2, 3, 4, 3>(simd2, &dense[0][0][0][0]); // Dims must be explicit this time
    value = 1_r;
    for (int i = 0; i < dense.dims[0]; ++i) {
      for (int j = 0; j < dense.dims[1]; ++j) {
        for (int k = 0; k < dense.dims[2]; ++k) {
          for (int l = 0; l < dense.dims[3]; ++l) {
            EXPECT_EQ(value, Get(simd1[i][j][k], l));
            EXPECT_EQ(value, Get(simd2[i][j][k], l));
            value += 1_r;
          }
        }
      }
    }
  }
}

TEST(BasicTypes, StoreMatrix) {
  // NdArray<real, 2, 4>
  {
    // Store 8 reals from SIMD with no padding
    NdArray<Vec4r, 2> simd = {Vec4r(1_r, 2_r, 3_r, 4_r), Vec4r(5_r, 6_r, 7_r, 8_r)};
    NdArray<real, 2, 4> dense1, dense2, dense3;
    StoreMatrix(dense1, simd);
    StoreMatrix<2, 4>(Flatten(dense2), simd); // Dims must be explicit this time
    StoreMatrix<2, 4>(&dense3[0][0], simd); // Dims must be explicit this time
    EXPECT_EQ(Real4(1_r, 2_r, 3_r, 4_r), dense1[0]);
    EXPECT_EQ(Real4(5_r, 6_r, 7_r, 8_r), dense1[1]);
    EXPECT_EQ(dense1, dense2);
    EXPECT_EQ(dense1, dense3);
  }

  // NdArray<real, 4, 3>
  {
    // Store 12 reals from SIMD which has padding.
    NdArray<Vec4r, 4> simd = {
        Vec4r(1_r, 2_r, 3_r, 911_r),
        Vec4r(4_r, 5_r, 6_r, 911_r),
        Vec4r(7_r, 8_r, 9_r, 911_r),
        Vec4r(10_r, 11_r, 12_r, 911_r)};
    NdArray<real, 4, 3> dense1, dense2, dense3;
    StoreMatrix(dense1, simd);
    StoreMatrix<4, 3>(Flatten(dense2), simd); // Dims must be explicit this time
    StoreMatrix<4, 3>(&dense3[0][0], simd); // Dims must be explicit this time
    EXPECT_EQ(Real3(1_r, 2_r, 3_r), dense1[0]);
    EXPECT_EQ(Real3(4_r, 5_r, 6_r), dense1[1]);
    EXPECT_EQ(Real3(7_r, 8_r, 9_r), dense1[2]);
    EXPECT_EQ(Real3(10_r, 11_r, 12_r), dense1[3]);
    EXPECT_EQ(dense1, dense2);
    EXPECT_EQ(dense1, dense3);
  }

  // NdArray<real, 2, 3, 4, 3>
  {
    NdArray<Vec4r, 2, 3, 4> simd;
    real value = 1_r;
    for (int i = 0; i < simd.dims[0]; ++i) {
      for (int j = 0; j < simd.dims[1]; ++j) {
        for (int k = 0; k < simd.dims[2]; ++k) {
          simd[i][j][k] = Vec4r(value, value + 0.4_r, value + 0.6_r, -1_r);
          value += 1_r;
        }
      }
    }

    NdArray<real, 2, 3, 4, 3> dense1, dense2;
    StoreMatrix(dense1, simd);
    StoreMatrix<2, 3, 4, 3>(&dense2[0][0][0][0], simd); // Dims must be explicit this time
    value = 1_r;
    for (int i = 0; i < simd.dims[0]; ++i) {
      for (int j = 0; j < simd.dims[1]; ++j) {
        for (int k = 0; k < simd.dims[2]; ++k) {
          EXPECT_EQ(Real3(value, value + 0.4_r, value + 0.6_r), dense1[i][j][k]);
          value += 1_r;
        }
      }
    }
    EXPECT_EQ(dense1, dense2);
  }
}

TEST(BasicTypes, SimdSymToFull) {
  Matrix3x3r T3{Real3{1.0_r, 4.0_r, 5.0_r}, Real3{4.0_r, 2.0_r, 6.0_r}, Real3{5.0_r, 6.0_r, 3.0_r}};
  VMatrix3x3r Tv3 = ToSimdMatrix(T3);
  VSymMatrix3x3r Tvsym3 = ToSimdSymMatrix(T3);
  VSymMatrix3x3r Tvsymtest3 = SimdFullToSym(Tv3);
  VMatrix3x3r Tvtest3 = SimdSymToFull(Tvsym3);
  EXPECT_EQ(Norm3x3(Tv3 - Tvtest3), 0_r);
  EXPECT_EQ(Norm<3>((Tvsym3 - Tvsymtest3)[0]), 0_r);
  EXPECT_EQ(Norm<3>((Tvsym3 - Tvsymtest3)[1]), 0_r);

  Matrix2x2r T2{Real2{1.0_r, 2.0_r}, Real2{2.0_r, 3.0_r}};
  VMatrix2x2r Tv2 = ToSimdMatrix(T2);
  VSymMatrix2x2r Tvsym2 = ToSimdSymMatrix(T2);
  VSymMatrix2x2r Tvsymtest2 = SimdFullToSym(Tv2);
  VMatrix2x2r Tvtest2 = SimdSymToFull(Tvsym2);
  EXPECT_EQ(Get0(VNorm<4>(Tv2 - Tvtest2)), 0_r);
  EXPECT_EQ(Get0(VNorm<3>(Tvsym2 - Tvsymtest2)), 0_r);
}

TEST(VMatrix, BroadcastEach_1D) {
  using V = Simd<real, 8>;
  auto pt = Real2{1_r, 2_r};
  NdArray<V, 2> vpt = BroadcastEach<V>(pt);
  EXPECT_EQ(V(1_r, 1_r, 1_r, 1_r, 1_r, 1_r, 1_r, 1_r), vpt[0]);
  EXPECT_EQ(V(2_r, 2_r, 2_r, 2_r, 2_r, 2_r, 2_r, 2_r), vpt[1]);
}

TEST(VMatrix, BroadcastEach_2D) {
  using V = Simd<real, 8>;
  auto mat = Matrix2x2r{Real2{1_r, 2_r}, Real2{3_r, 4_r}};
  NdArray<V, 2, 2> vmat = BroadcastEach<V>(mat);
  EXPECT_EQ(V(1_r, 1_r, 1_r, 1_r, 1_r, 1_r, 1_r, 1_r), vmat[0][0]);
  EXPECT_EQ(V(2_r, 2_r, 2_r, 2_r, 2_r, 2_r, 2_r, 2_r), vmat[0][1]);
  EXPECT_EQ(V(3_r, 3_r, 3_r, 3_r, 3_r, 3_r, 3_r, 3_r), vmat[1][0]);
  EXPECT_EQ(V(4_r, 4_r, 4_r, 4_r, 4_r, 4_r, 4_r, 4_r), vmat[1][1]);
}

TEST(VMatrix, BroadcastEach_Simd) {
  {
    using V = Vec4r;
    auto pt = V{1_r, 2_r, 3_r, 4_r};
    NdArray<V, 4> vpt = BroadcastEach<V>(pt);
    EXPECT_EQ(V(1_r, 1_r, 1_r, 1_r), vpt[0]);
    EXPECT_EQ(V(2_r, 2_r, 2_r, 2_r), vpt[1]);
    EXPECT_EQ(V(3_r, 3_r, 3_r, 3_r), vpt[2]);
    EXPECT_EQ(V(4_r, 4_r, 4_r, 4_r), vpt[3]);
  }
  {
    using V = Simd<real, 8>;
    auto pt = Vec4r{1_r, 2_r, 3_r, 4_r}; // Not the same size as V
    NdArray<V, 4> vpt = BroadcastEach<V>(pt);
    EXPECT_EQ(V(1_r, 1_r, 1_r, 1_r, 1_r, 1_r, 1_r, 1_r), vpt[0]);
    EXPECT_EQ(V(2_r, 2_r, 2_r, 2_r, 2_r, 2_r, 2_r, 2_r), vpt[1]);
    EXPECT_EQ(V(3_r, 3_r, 3_r, 3_r, 3_r, 3_r, 3_r, 3_r), vpt[2]);
    EXPECT_EQ(V(4_r, 4_r, 4_r, 4_r, 4_r, 4_r, 4_r, 4_r), vpt[3]);
  }
}

TEST(VMatrix, Broadcast3) {
  {
    using V = Vec4r;
    auto pt = Vec4r{1_r, 2_r, 3_r, 4_r};
    NdArray<V, 3> vpt = Broadcast3<V>(pt);
    EXPECT_EQ(V(1_r, 1_r, 1_r, 1_r), vpt[0]);
    EXPECT_EQ(V(2_r, 2_r, 2_r, 2_r), vpt[1]);
    EXPECT_EQ(V(3_r, 3_r, 3_r, 3_r), vpt[2]);
  }
  {
    using V = Vec8r;
    auto pt = Vec4r{1_r, 2_r, 3_r, 4_r}; // Not the same size as V
    NdArray<V, 3> vpt = Broadcast3<V>(pt);
    EXPECT_EQ(V(1_r, 1_r, 1_r, 1_r, 1_r, 1_r, 1_r, 1_r), vpt[0]);
    EXPECT_EQ(V(2_r, 2_r, 2_r, 2_r, 2_r, 2_r, 2_r, 2_r), vpt[1]);
    EXPECT_EQ(V(3_r, 3_r, 3_r, 3_r, 3_r, 3_r, 3_r, 3_r), vpt[2]);
  }
}

TEST(VMatrix, Broadcast3x3) {
  using V = Vec8r;
  auto mat = VMatrix4x4r{
      Vec4r{1_r, 2_r, 3_r, 4_r},
      Vec4r{5_r, 6_r, 7_r, 8_r},
      Vec4r{9_r, 10_r, 11_r, 12_r},
      Vec4r{13_r, 14_r, 15_r, 16_r}};
  NdArray<V, 3, 3> vmat = Broadcast3x3<V>(mat);
  EXPECT_EQ(V(1_r, 1_r, 1_r, 1_r, 1_r, 1_r, 1_r, 1_r), vmat[0][0]);
  EXPECT_EQ(V(2_r, 2_r, 2_r, 2_r, 2_r, 2_r, 2_r, 2_r), vmat[0][1]);
  EXPECT_EQ(V(3_r, 3_r, 3_r, 3_r, 3_r, 3_r, 3_r, 3_r), vmat[0][2]);
  EXPECT_EQ(V(5_r, 5_r, 5_r, 5_r, 5_r, 5_r, 5_r, 5_r), vmat[1][0]);
  EXPECT_EQ(V(6_r, 6_r, 6_r, 6_r, 6_r, 6_r, 6_r, 6_r), vmat[1][1]);
  EXPECT_EQ(V(7_r, 7_r, 7_r, 7_r, 7_r, 7_r, 7_r, 7_r), vmat[1][2]);
  EXPECT_EQ(V(9_r, 9_r, 9_r, 9_r, 9_r, 9_r, 9_r, 9_r), vmat[2][0]);
  EXPECT_EQ(V(10_r, 10_r, 10_r, 10_r, 10_r, 10_r, 10_r, 10_r), vmat[2][1]);
  EXPECT_EQ(V(11_r, 11_r, 11_r, 11_r, 11_r, 11_r, 11_r, 11_r), vmat[2][2]);
}

TEST(VMatrix, LoadStoreTransposed) {
  constexpr std::array<real, 24> kValues{1_r,  2_r,  3_r,  4_r,  5_r,  6_r,  7_r,  8_r,
                                         9_r,  10_r, 11_r, 12_r, 13_r, 14_r, 15_r, 16_r,
                                         17_r, 18_r, 19_r, 20_r, 21_r, 22_r, 23_r, 24_r};
  NdArray<Vec8r, 3> v;
  LoadTransposed<1>(kValues.data(), v);
  EXPECT_EQ(Vec8r(1_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r), v[0]);
  EXPECT_EQ(Vec8r(2_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r), v[1]);
  EXPECT_EQ(Vec8r(3_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r), v[2]);
  LoadTransposed<2>(kValues.data(), v);
  EXPECT_EQ(Vec8r(1_r, 4_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r), v[0]);
  EXPECT_EQ(Vec8r(2_r, 5_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r), v[1]);
  EXPECT_EQ(Vec8r(3_r, 6_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r), v[2]);
  LoadTransposed<3>(kValues.data(), v);
  EXPECT_EQ(Vec8r(1_r, 4_r, 7_r, 0_r, 0_r, 0_r, 0_r, 0_r), v[0]);
  EXPECT_EQ(Vec8r(2_r, 5_r, 8_r, 0_r, 0_r, 0_r, 0_r, 0_r), v[1]);
  EXPECT_EQ(Vec8r(3_r, 6_r, 9_r, 0_r, 0_r, 0_r, 0_r, 0_r), v[2]);
  LoadTransposed<4>(kValues.data(), v);
  EXPECT_EQ(Vec8r(1_r, 4_r, 7_r, 10_r, 0_r, 0_r, 0_r, 0_r), v[0]);
  EXPECT_EQ(Vec8r(2_r, 5_r, 8_r, 11_r, 0_r, 0_r, 0_r, 0_r), v[1]);
  EXPECT_EQ(Vec8r(3_r, 6_r, 9_r, 12_r, 0_r, 0_r, 0_r, 0_r), v[2]);
  LoadTransposed<5>(kValues.data(), v);
  EXPECT_EQ(Vec8r(1_r, 4_r, 7_r, 10_r, 13_r, 0_r, 0_r, 0_r), v[0]);
  EXPECT_EQ(Vec8r(2_r, 5_r, 8_r, 11_r, 14_r, 0_r, 0_r, 0_r), v[1]);
  EXPECT_EQ(Vec8r(3_r, 6_r, 9_r, 12_r, 15_r, 0_r, 0_r, 0_r), v[2]);
  LoadTransposed<6>(kValues.data(), v);
  EXPECT_EQ(Vec8r(1_r, 4_r, 7_r, 10_r, 13_r, 16_r, 0_r, 0_r), v[0]);
  EXPECT_EQ(Vec8r(2_r, 5_r, 8_r, 11_r, 14_r, 17_r, 0_r, 0_r), v[1]);
  EXPECT_EQ(Vec8r(3_r, 6_r, 9_r, 12_r, 15_r, 18_r, 0_r, 0_r), v[2]);
  LoadTransposed<7>(kValues.data(), v);
  EXPECT_EQ(Vec8r(1_r, 4_r, 7_r, 10_r, 13_r, 16_r, 19_r, 0_r), v[0]);
  EXPECT_EQ(Vec8r(2_r, 5_r, 8_r, 11_r, 14_r, 17_r, 20_r, 0_r), v[1]);
  EXPECT_EQ(Vec8r(3_r, 6_r, 9_r, 12_r, 15_r, 18_r, 21_r, 0_r), v[2]);
  LoadTransposed(kValues.data(), v);
  EXPECT_EQ(Vec8r(1_r, 4_r, 7_r, 10_r, 13_r, 16_r, 19_r, 22_r), v[0]);
  EXPECT_EQ(Vec8r(2_r, 5_r, 8_r, 11_r, 14_r, 17_r, 20_r, 23_r), v[1]);
  EXPECT_EQ(Vec8r(3_r, 6_r, 9_r, 12_r, 15_r, 18_r, 21_r, 24_r), v[2]);

  auto testStoreTransposed = [&]<int N>(std::integral_constant<int, N>) {
    std::array<real, 25> storedValues{};
    storedValues[24] = 911_r; // sentinel
    StoreTransposed<N>(storedValues.data(), v);
    for (int i = 0; i < isize(kValues); ++i) {
      EXPECT_EQ(i < 3 * N ? kValues[i] : 0_r, storedValues[i]);
    }
    EXPECT_EQ(911_r, storedValues[24]); // no change
  };

  testStoreTransposed(std::integral_constant<int, 1>{});
  testStoreTransposed(std::integral_constant<int, 2>{});
  testStoreTransposed(std::integral_constant<int, 3>{});
  testStoreTransposed(std::integral_constant<int, 4>{});
  testStoreTransposed(std::integral_constant<int, 5>{});
  testStoreTransposed(std::integral_constant<int, 6>{});
  testStoreTransposed(std::integral_constant<int, 7>{});
  testStoreTransposed(std::integral_constant<int, 8>{});
}
