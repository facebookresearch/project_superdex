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

#include <mochi_core/linear_algebra/any_matrix.h>
#include <mochi_core/linear_algebra/any_matrix_utils.h>
#include <mochi_core/linear_algebra/block_sparse_matrix.h>
#include <mochi_core/linear_algebra/block_sparse_matrix_utils.h>
#include <mochi_core/linear_algebra/matrix_utils.h>
#include <mochi_core/linear_algebra/sparse_matrix.h>
#include <mochi_core/linear_algebra/sparse_matrix_utils.h>
#include <mochi_core/linear_algebra/utils/matrix_conversions.h>
#include <mochi_core/mochi_config.h>
#include <mochi_core/solvers/island_operators.h>
#include <mochi_core/solvers/island_operators_utils.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/nd_array_utils.h>

#include <type_traits>

using namespace mochi;
using namespace mochi::test;

TEST(MatrixConversion, FromMatrix) {
  // Empty
  {
    RowMatrix<float> src;
    EXPECT_TRUE(NearEqualMatrices(src, ToMatrix(src)));
    EXPECT_TRUE(NearEqualMatrices(src, ToMatrix(AsConstView(src))));
    EXPECT_TRUE(NearEqualMatrices(src, ToSparseMatrix(src)));
    EXPECT_TRUE(NearEqualMatrices(src, ToSparseMatrix(AsConstView(src))));
    EXPECT_TRUE(NearEqualMatrices(src, ToBlockSparseMatrix<1>(src)));
    EXPECT_TRUE(NearEqualMatrices(src, ToBlockSparseMatrix<1>(AsConstView(src))));
    EXPECT_TRUE(NearEqualMatrices(src, ToBlockSparseMatrix<3>(src)));
    EXPECT_TRUE(NearEqualMatrices(src, ToBlockSparseMatrix<3>(AsConstView(src))));
  }

  // Fixed RowVector
  {
    RowVector<float, 7> src;
    src.SetRandom(123);
    EXPECT_TRUE(NearEqualMatrices(src, ToMatrix(src)));
    EXPECT_TRUE(NearEqualMatrices(src, ToMatrix(AsConstView(src))));
    EXPECT_TRUE(NearEqualMatrices(src, ToSparseMatrix(src)));
    EXPECT_TRUE(NearEqualMatrices(src, ToSparseMatrix(AsConstView(src))));
    EXPECT_TRUE(NearEqualMatrices(src, ToBlockSparseMatrix<1>(src)));
    EXPECT_TRUE(NearEqualMatrices(src, ToBlockSparseMatrix<1>(AsConstView(src))));
  }

  // Dynamic RowVector
  {
    RowVector<float> src(11);
    src.SetRandom(123);
    EXPECT_TRUE(NearEqualMatrices(src, ToMatrix(src)));
    EXPECT_TRUE(NearEqualMatrices(src, ToMatrix(AsConstView(src))));
    EXPECT_TRUE(NearEqualMatrices(src, ToSparseMatrix(src)));
    EXPECT_TRUE(NearEqualMatrices(src, ToSparseMatrix(AsConstView(src))));
    EXPECT_TRUE(NearEqualMatrices(src, ToBlockSparseMatrix<1>(src)));
    EXPECT_TRUE(NearEqualMatrices(src, ToBlockSparseMatrix<1>(AsConstView(src))));
  }

  // Fixed RowMatrix
  {
    RowMatrix<float, 9, 12> src;
    src.SetRandom(123);
    EXPECT_TRUE(NearEqualMatrices(src, ToMatrix(src)));
    EXPECT_TRUE(NearEqualMatrices(src, ToMatrix(AsConstView(src))));
    EXPECT_TRUE(NearEqualMatrices(src, ToSparseMatrix(src)));
    EXPECT_TRUE(NearEqualMatrices(src, ToSparseMatrix(AsConstView(src))));
    EXPECT_TRUE(NearEqualMatrices(src, ToSparseMatrix(src, true)));
    EXPECT_TRUE(NearEqualMatrices(src, ToSparseMatrix(AsConstView(src), true)));
    EXPECT_TRUE(NearEqualMatrices(src, ToBlockSparseMatrix<1>(src)));
    EXPECT_TRUE(NearEqualMatrices(src, ToBlockSparseMatrix<1>(AsConstView(src))));
    EXPECT_TRUE(NearEqualMatrices(src, ToBlockSparseMatrix<1>(src, true)));
    EXPECT_TRUE(NearEqualMatrices(src, ToBlockSparseMatrix<1>(AsConstView(src), true)));
    EXPECT_TRUE(NearEqualMatrices(src, ToBlockSparseMatrix<3>(src)));
    EXPECT_TRUE(NearEqualMatrices(src, ToBlockSparseMatrix<3>(AsConstView(src))));
    EXPECT_TRUE(NearEqualMatrices(src, ToBlockSparseMatrix<3>(src, true)));
    EXPECT_TRUE(NearEqualMatrices(src, ToBlockSparseMatrix<3>(AsConstView(src), true)));
  }

  // Dynamic RowMatrix
  {
    RowMatrix<float> src(12, 15);
    src.SetRandom(123);
    EXPECT_TRUE(NearEqualMatrices(src, ToMatrix(src)));
    EXPECT_TRUE(NearEqualMatrices(src, ToMatrix(AsConstView(src))));
    EXPECT_TRUE(NearEqualMatrices(src, ToSparseMatrix(src)));
    EXPECT_TRUE(NearEqualMatrices(src, ToSparseMatrix(AsConstView(src))));
    EXPECT_TRUE(NearEqualMatrices(src, ToSparseMatrix(src, true)));
    EXPECT_TRUE(NearEqualMatrices(src, ToSparseMatrix(AsConstView(src), true)));
    EXPECT_TRUE(NearEqualMatrices(src, ToBlockSparseMatrix<1>(src)));
    EXPECT_TRUE(NearEqualMatrices(src, ToBlockSparseMatrix<1>(AsConstView(src))));
    EXPECT_TRUE(NearEqualMatrices(src, ToBlockSparseMatrix<1>(src, true)));
    EXPECT_TRUE(NearEqualMatrices(src, ToBlockSparseMatrix<1>(AsConstView(src), true)));
    EXPECT_TRUE(NearEqualMatrices(src, ToBlockSparseMatrix<3>(src)));
    EXPECT_TRUE(NearEqualMatrices(src, ToBlockSparseMatrix<3>(AsConstView(src))));
    EXPECT_TRUE(NearEqualMatrices(src, ToBlockSparseMatrix<3>(src, true)));
    EXPECT_TRUE(NearEqualMatrices(src, ToBlockSparseMatrix<3>(AsConstView(src), true)));
  }

  {
    using Scalar = float;
    auto C = Matrix<Scalar, 6, 6>::Zero(); // Col-major, compile-time
    C(0, 0) = Scalar(2);
    C(0, 1) = Scalar(-1);
    C(1, 1) = Scalar(3);
    C(1, 2) = Scalar(-1);
    C(2, 2) = Scalar(4);
    C(2, 3) = Scalar(-1);
    C(3, 3) = Scalar(2);
    C(3, 4) = Scalar(-1);
    C(4, 4) = Scalar(2);
    C(4, 5) = Scalar(-1);
    C(5, 5) = Scalar(2);
    for (int ii = 0; ii < 6; ++ii) {
      for (int jj = 0; jj < ii; ++jj) {
        C(ii, jj) = C(jj, ii);
      }
    }

    //-- Conversion to BlockSparseMatrix with block size of 1
    {
      //-- With pruning
      auto bSpC = ToBlockSparseMatrix<1>(C, true);
      EXPECT_TRUE(NearEqualMatrices(C, bSpC));
      EXPECT_EQ(16, bSpC.NumNonZeroBlocks());
    }
    {
      //-- Without pruning
      auto bSpC = ToBlockSparseMatrix<1>(C);
      EXPECT_TRUE(NearEqualMatrices(C, bSpC));
      EXPECT_EQ(36, bSpC.NumNonZeroBlocks());
    }
    EXPECT_TRUE(NearEqualMatrices(C, ToBlockSparseMatrix<1>(AsConstView(C))));

    //-- Conversion to BlockSparseMatrix with block size of 2
    {
      //-- With pruning
      auto bSpC = ToBlockSparseMatrix<2>(C, true);
      EXPECT_TRUE(NearEqualMatrices(C, bSpC));
      EXPECT_EQ(7, bSpC.NumNonZeroBlocks());
    }
    {
      //-- Without pruning
      auto bSpC = ToBlockSparseMatrix<2>(C);
      EXPECT_TRUE(NearEqualMatrices(C, bSpC));
      EXPECT_EQ(9, bSpC.NumNonZeroBlocks());
    }
    EXPECT_TRUE(NearEqualMatrices(C, ToBlockSparseMatrix<2>(AsConstView(C))));

    //-- Conversion to BlockSparseMatrix with block size of 3
    {
      //-- With pruning
      auto bSpC = ToBlockSparseMatrix<3>(C, true);
      EXPECT_TRUE(NearEqualMatrices(C, bSpC));
      EXPECT_EQ(4, bSpC.NumNonZeroBlocks());
    }
    {
      //-- Without pruning
      auto bSpC = ToBlockSparseMatrix<3>(C);
      EXPECT_TRUE(NearEqualMatrices(C, bSpC));
      EXPECT_EQ(4, bSpC.NumNonZeroBlocks());
    }
    EXPECT_TRUE(NearEqualMatrices(C, ToBlockSparseMatrix<3>(AsConstView(C))));

    //-- Conversion to BlockSparseMatrix with block size of 6
    {
      //-- With pruning
      auto bSpC = ToBlockSparseMatrix<6>(C, true);
      EXPECT_TRUE(NearEqualMatrices(C, bSpC));
      EXPECT_EQ(1, bSpC.NumNonZeroBlocks());
    }
    {
      //-- Without pruning
      auto bSpC = ToBlockSparseMatrix<6>(C);
      EXPECT_TRUE(NearEqualMatrices(C, bSpC));
      EXPECT_EQ(1, bSpC.NumNonZeroBlocks());
    }
    EXPECT_TRUE(NearEqualMatrices(C, ToBlockSparseMatrix<6>(AsConstView(C))));

    //-- Conversion to Matrix
    EXPECT_TRUE(NearEqualMatrices(C, ToMatrix(C)));
    EXPECT_TRUE(NearEqualMatrices(C, ToMatrix(AsConstView(C))));

    //-- Conversion to SparseMatrix
    {
      //-- With pruning
      auto spC = ToSparseMatrix(C, true);
      EXPECT_TRUE(NearEqualMatrices(C, spC));
      EXPECT_EQ(16, spC.NumNonZeros());
    }
    {
      //-- Without pruning
      auto spC = ToSparseMatrix(C);
      EXPECT_TRUE(NearEqualMatrices(C, spC));
      EXPECT_EQ(36, spC.NumNonZeros());
    }
    EXPECT_TRUE(NearEqualMatrices(C, ToSparseMatrix(AsConstView(C))));
  }
}

TEST(MatrixConversion, FromBlockSparseMatrix) {
  // Empty
  {
    BlockSparseMatrix<float, 3> src;
    EXPECT_TRUE(NearEqualMatrices(src, ToMatrix(src)));
    EXPECT_TRUE(NearEqualMatrices(src, ToMatrix(AsConstView(src))));
    EXPECT_TRUE(NearEqualMatrices(src, ToSparseMatrix(src)));
    EXPECT_TRUE(NearEqualMatrices(src, ToSparseMatrix(AsConstView(src))));
    EXPECT_TRUE(NearEqualMatrices(src, ToBlockSparseMatrix<1>(src)));
    EXPECT_TRUE(NearEqualMatrices(src, ToBlockSparseMatrix<1>(AsConstView(src))));
    EXPECT_TRUE(NearEqualMatrices(src, ToBlockSparseMatrix<3>(src)));
    EXPECT_TRUE(NearEqualMatrices(src, ToBlockSparseMatrix<3>(AsConstView(src))));
    EXPECT_TRUE(NearEqualMatrices(src, ToBlockSparseMatrix<6>(src)));
    EXPECT_TRUE(NearEqualMatrices(src, ToBlockSparseMatrix<6>(AsConstView(src))));
  }

  // BlockSparse 15 x 12
  {
    // |***   ***   |
    // |***   ***   |
    // |***   ***   |
    // |   ***   ***|
    // |   ***   ***|
    // |   ***   ***|
    // |******      |
    // |******      |
    // |******      |
    // |   *********|
    // |   *********|
    // |   *********|
    // |************|
    // |************|
    // |************|
    int constexpr kNumNonZeroBlocks = 13;
    ColumnVector<float> denseValues(kNumNonZeroBlocks * 3 * 3);
    denseValues.SetRandom(123);

    {
      // kBlockSize == 1
      // clang-format off
    BlockSparseMatrix<float, 1> src{
        12,
        DynamicArray<int>{0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 63, 72, 81, 93, 105, 117},
        DynamicArray<int>{0, 1, 2, 6, 7,  8,  0, 1,  2,  6, 7,  8,  0, 1, 2, 6, 7,  8,
                         3, 4, 5, 9, 10, 11, 3, 4,  5,  9, 10, 11, 3, 4, 5, 9, 10, 11,
                         0, 1, 2, 3, 4,  5,  0, 1,  2,  3, 4,  5,  0, 1, 2, 3, 4,  5,
                         3, 4, 5, 6, 7,  8,  9, 10, 11, 3, 4,  5,  6, 7, 8, 9, 10, 11, 3, 4, 5, 6, 7,  8,  9, 10, 11,
                         0, 1, 2, 3, 4,  5,  6, 7,  8,  9, 10, 11, 0, 1, 2, 3, 4,  5,  6, 7, 8, 9, 10, 11, 0, 1,  2,  3, 4,  5,  6, 7, 8, 9, 10, 11},
        DynamicArray<float>(denseValues.GetConstSpan().begin(), denseValues.GetConstSpan().end())};

    auto view1 = ToBlockSparseMatrix<1>(src);
    auto view2 = ToBlockSparseMatrix<1>(AsView(src));
    auto view3 = ToBlockSparseMatrix<1>(AsConstView(src));

    // All should be const views of the same type
    static_assert(std::is_const_v<typename decltype(view1)::Scalar>, "Expected const view");
    static_assert(std::is_same_v<decltype(view1), decltype(view2)>, "Expected same view type");
    static_assert(std::is_same_v<decltype(view1), decltype(view3)>, "Expected same view type");

    // All should be equivalent
    EXPECT_TRUE(NearEqualMatrices(src, view1));
    EXPECT_TRUE(NearEqualMatrices(src, view2));
    EXPECT_TRUE(NearEqualMatrices(src, view3));
    EXPECT_TRUE(NearEqualMatrices(src, ToBlockSparseMatrix<3>(view1))); // Also with block size of 3

    // In fact, they should point to the same data
    EXPECT_EQ(src.Values().data(), view1.Values().data());
    EXPECT_EQ(src.Values().data(), view2.Values().data());
    EXPECT_EQ(src.Values().data(), view3.Values().data());

    // Compare as dense matrices
    EXPECT_TRUE(NearEqualMatrices(src, ToMatrix(src)));
    EXPECT_TRUE(NearEqualMatrices(src, ToMatrix(AsConstView(src))));

    // Compare as sparse matrices
    EXPECT_TRUE(NearEqualMatrices(src, ToSparseMatrix(src)));
    EXPECT_TRUE(NearEqualMatrices(src, ToSparseMatrix(AsConstView(src))));
      // clang-format on
    }

    {
      // kBlockSize == 3
      BlockSparseMatrix<float, 3> src{
          4,
          DynamicArray<int>{0, 2, 4, 6, 9, 13},
          DynamicArray<int>{0, 2, 1, 3, 0, 1, 1, 2, 3, 0, 1, 2, 3},
          DynamicArray<float>(
              denseValues.GetConstSpan().begin(), denseValues.GetConstSpan().end())};

      auto view1 = ToBlockSparseMatrix<3>(src);
      auto view2 = ToBlockSparseMatrix<3>(AsView(src));
      auto view3 = ToBlockSparseMatrix<3>(AsConstView(src));

      // All should be const views of the same type
      static_assert(std::is_const_v<typename decltype(view1)::Scalar>, "Expected const view");
      static_assert(std::is_same_v<decltype(view1), decltype(view2)>, "Expected same view type");
      static_assert(std::is_same_v<decltype(view1), decltype(view3)>, "Expected same view type");

      // All should be equivalent
      EXPECT_TRUE(NearEqualMatrices(src, view1));
      EXPECT_TRUE(NearEqualMatrices(src, view2));
      EXPECT_TRUE(NearEqualMatrices(src, view3));
      EXPECT_TRUE(
          NearEqualMatrices(src, ToBlockSparseMatrix<1>(view1))); // Also with block size of 1

      // In fact, they should point to the same data
      EXPECT_EQ(src.Values().data(), view1.Values().data());
      EXPECT_EQ(src.Values().data(), view2.Values().data());
      EXPECT_EQ(src.Values().data(), view3.Values().data());

      // Compare as dense matrices
      EXPECT_TRUE(NearEqualMatrices(src, ToMatrix(src)));
      EXPECT_TRUE(NearEqualMatrices(src, ToMatrix(AsConstView(src))));

      // Compare as sparse matrices
      EXPECT_TRUE(NearEqualMatrices(src, ToSparseMatrix(src)));
      EXPECT_TRUE(NearEqualMatrices(src, ToSparseMatrix(AsConstView(src))));
    }
  }
}

TEST(MatrixConversions, FromSparseMatrix) {
  // |1 2|
  // | 3 |
  // |4 5|
  SparseMatrix<float> src(
      3,
      DynamicArray<int>{0, 2, 3, 5},
      DynamicArray<int>{0, 2, 1, 0, 2},
      DynamicArray<float>{1_r, 2_r, 3_r, 4_r, 5_r});

  auto view1 = ToSparseMatrix(src);
  auto view2 = ToSparseMatrix(AsView(src));
  auto view3 = ToSparseMatrix(AsConstView(src));

  // All three should be const views of the same type
  static_assert(std::is_const_v<typename decltype(view1)::Scalar>, "Expected const view");
  static_assert(std::is_same_v<decltype(view1), decltype(view2)>, "Expected same view type");
  static_assert(std::is_same_v<decltype(view1), decltype(view3)>, "Expected same view type");

  // All three should be equivalent
  EXPECT_TRUE(NearEqualMatrices(src, ToSparseMatrix(view1)));
  EXPECT_TRUE(NearEqualMatrices(src, ToSparseMatrix(view2)));
  EXPECT_TRUE(NearEqualMatrices(src, ToSparseMatrix(view3)));

  // In fact, they should point to the same data
  EXPECT_EQ(view1.Values().data(), view2.Values().data());
  EXPECT_EQ(view1.Values().data(), view3.Values().data());

  // Compare as dense matrices
  EXPECT_TRUE(NearEqualMatrices(src, ToMatrix(view1)));
  EXPECT_TRUE(NearEqualMatrices(src, ToMatrix(view2)));
  EXPECT_TRUE(NearEqualMatrices(src, ToMatrix(view3)));

  // Compare as block sparse matrices
  EXPECT_TRUE(NearEqualMatrices(src, ToBlockSparseMatrix<1>(view1)));
  EXPECT_TRUE(NearEqualMatrices(src, ToBlockSparseMatrix<1>(view2)));
  EXPECT_TRUE(NearEqualMatrices(src, ToBlockSparseMatrix<1>(view3)));
  EXPECT_TRUE(NearEqualMatrices(src, ToBlockSparseMatrix<3>(view1)));
  EXPECT_TRUE(NearEqualMatrices(src, ToBlockSparseMatrix<3>(view2)));
  EXPECT_TRUE(NearEqualMatrices(src, ToBlockSparseMatrix<3>(view3)));
}

#if MOCHI_USE_CUDA
TEST(MatrixConversion, ToCuda) {
  using Scalar = float;

  // Empty
  {
    Matrix<Scalar> src1;
    EXPECT_TRUE(NearEqualMatrices(src1, Matrix<Scalar>(ToCuda(src1))));
    EXPECT_TRUE(NearEqualMatrices(src1, Matrix<Scalar>(ToCuda(AsView(src1)))));
    EXPECT_TRUE(NearEqualMatrices(src1, Matrix<Scalar>(ToCuda(AsConstView(src1)))));

    RowMatrix<Scalar> src2;
    EXPECT_TRUE(NearEqualMatrices(src2, RowMatrix<Scalar>(ToCuda(src2))));
    EXPECT_TRUE(NearEqualMatrices(src2, RowMatrix<Scalar>(ToCuda(AsView(src2)))));
    EXPECT_TRUE(NearEqualMatrices(src2, RowMatrix<Scalar>(ToCuda(AsConstView(src2)))));
  }

  // Compile-time vector
  {
    ColumnVector<Scalar, 7> src1;
    src1.SetRandom(1);
    EXPECT_TRUE(NearEqualMatrices(src1, Matrix<Scalar>(ToCuda(src1))));
    EXPECT_TRUE(NearEqualMatrices(src1, Matrix<Scalar>(ToCuda(AsView(src1)))));
    EXPECT_TRUE(NearEqualMatrices(src1, Matrix<Scalar>(ToCuda(AsConstView(src1)))));

    RowVector<Scalar, 12> src2;
    src2.SetRandom(2);
    EXPECT_TRUE(NearEqualMatrices(src2, RowMatrix<Scalar>(ToCuda(src2))));
    EXPECT_TRUE(NearEqualMatrices(src2, RowMatrix<Scalar>(ToCuda(AsView(src2)))));
    EXPECT_TRUE(NearEqualMatrices(src2, RowMatrix<Scalar>(ToCuda(AsConstView(src2)))));
  }

  // Dynamic vector
  {
    ColumnVector<Scalar> src1(11);
    src1.SetRandom(3);
    EXPECT_TRUE(NearEqualMatrices(src1, Matrix<Scalar>(ToCuda(src1))));
    EXPECT_TRUE(NearEqualMatrices(src1, Matrix<Scalar>(ToCuda(AsView(src1)))));
    EXPECT_TRUE(NearEqualMatrices(src1, Matrix<Scalar>(ToCuda(AsConstView(src1)))));

    RowVector<Scalar> src2(21);
    src2.SetRandom(4);
    EXPECT_TRUE(NearEqualMatrices(src2, RowMatrix<Scalar>(ToCuda(src2))));
    EXPECT_TRUE(NearEqualMatrices(src2, RowMatrix<Scalar>(ToCuda(AsView(src2)))));
    EXPECT_TRUE(NearEqualMatrices(src2, RowMatrix<Scalar>(ToCuda(AsConstView(src2)))));
  }

  // Compile-time matrix
  {
    Matrix<Scalar, 9, 12> src1;
    src1.SetRandom(5);
    EXPECT_TRUE(NearEqualMatrices(src1, Matrix<Scalar>(ToCuda(src1))));
    EXPECT_TRUE(NearEqualMatrices(src1, Matrix<Scalar>(ToCuda(AsView(src1)))));
    EXPECT_TRUE(NearEqualMatrices(src1, Matrix<Scalar>(ToCuda(AsConstView(src1)))));

    RowMatrix<Scalar, 11, 7> src2;
    src2.SetRandom(6);
    EXPECT_TRUE(NearEqualMatrices(src2, RowMatrix<Scalar>(ToCuda(src2))));
    EXPECT_TRUE(NearEqualMatrices(src2, RowMatrix<Scalar>(ToCuda(AsView(src2)))));
    EXPECT_TRUE(NearEqualMatrices(src2, RowMatrix<Scalar>(ToCuda(AsConstView(src2)))));
  }

  // Dynamic matrix
  {
    Matrix<Scalar> src1(9, 21);
    src1.SetRandom(7);
    EXPECT_TRUE(NearEqualMatrices(src1, Matrix<Scalar>(ToCuda(src1))));
    EXPECT_TRUE(NearEqualMatrices(src1, Matrix<Scalar>(ToCuda(AsView(src1)))));
    EXPECT_TRUE(NearEqualMatrices(src1, Matrix<Scalar>(ToCuda(AsConstView(src1)))));

    RowMatrix<Scalar> src2(12, 15);
    src2.SetRandom(8);
    EXPECT_TRUE(NearEqualMatrices(src2, RowMatrix<Scalar>(ToCuda(src2))));
    EXPECT_TRUE(NearEqualMatrices(src2, RowMatrix<Scalar>(ToCuda(AsView(src2)))));
    EXPECT_TRUE(NearEqualMatrices(src2, RowMatrix<Scalar>(ToCuda(AsConstView(src2)))));
  }
}
#endif // MOCHI_USE_CUDA

// TODO(@pabfer): Extend unit tests coverage. Include different Matrix specializations, including
// leading dimension larger than size.

TEST(StaticCast, BlockSparseMatrix) {
  float constexpr fTol = std::numeric_limits<float>::epsilon();
  double constexpr dTol = std::numeric_limits<double>::epsilon();
  // Test casting BlockSparseMatrix from float to double
  {
    int const numBlocks = 6;
    DynamicArray<float> val(numBlocks * 3 * 3);
    ColumnVectorView<float> valView(val.data(), isize(val));
    valView.SetRandom(123);
    BlockSparseMatrix<float, 3> srcFloat{
        4, DynamicArray<int>{0, 2, 4, 6}, DynamicArray<int>{0, 2, 1, 3, 0, 1}, std::move(val)};

    auto castDouble = StaticCast<BlockSparseMatrix<double, 3>>(srcFloat);

    EXPECT_EQ(srcFloat.BlockRows(), castDouble.BlockRows());
    EXPECT_EQ(srcFloat.BlockCols(), castDouble.BlockCols());
    EXPECT_EQ(srcFloat.NumNonZeroBlocks(), castDouble.NumNonZeroBlocks());
    EXPECT_TRUE(NearEqualSpan(srcFloat.Pointers(), castDouble.Pointers()));
    EXPECT_TRUE(NearEqualSpan(srcFloat.Indices(), castDouble.Indices()));
    EXPECT_TRUE(NearEqualSpan(srcFloat.Values(), castDouble.Values(), fTol));

    // Verify type conversion
    static_assert(
        std::is_same_v<typename decltype(castDouble)::Scalar, double>,
        "Expected double scalar type");
  }

  // Test casting BlockSparseMatrix from double to float
  {
    int const numBlocks = 3;
    DynamicArray<double> val(numBlocks * 2 * 2);
    ColumnVectorView<double> valView(val.data(), isize(val));
    valView.SetRandom(123);
    BlockSparseMatrix<double, 2> srcDouble{
        3, DynamicArray<int>{0, 2, 3}, DynamicArray<int>{0, 1, 2}, std::move(val)};

    auto castFloat = StaticCast<BlockSparseMatrix<float, 2>>(srcDouble);

    EXPECT_EQ(srcDouble.BlockRows(), castFloat.BlockRows());
    EXPECT_EQ(srcDouble.BlockCols(), castFloat.BlockCols());
    EXPECT_EQ(srcDouble.NumNonZeroBlocks(), castFloat.NumNonZeroBlocks());
    EXPECT_TRUE(NearEqualSpan(srcDouble.Pointers(), castFloat.Pointers()));
    EXPECT_TRUE(NearEqualSpan(srcDouble.Indices(), castFloat.Indices()));
    EXPECT_TRUE(NearEqualSpan(castFloat.Values(), srcDouble.Values(), fTol));
    // Check that some data is lost in the conversion
    EXPECT_FALSE(NearEqualSpan(srcDouble.Values(), castFloat.Values(), dTol));

    // Verify type conversion
    static_assert(
        std::is_same_v<typename decltype(castFloat)::Scalar, float>, "Expected float scalar type");
  }

  // Test with empty matrix
  {
    BlockSparseMatrix<float, 1> srcEmpty;
    auto castEmpty = StaticCast<BlockSparseMatrix<double, 1>>(srcEmpty);
    EXPECT_EQ(0, castEmpty.BlockRows());
    EXPECT_EQ(0, castEmpty.BlockCols());
    EXPECT_EQ(0, castEmpty.NumNonZeroBlocks());
  }
}

TEST(StaticCast, Matrix) {
  float constexpr fTol = std::numeric_limits<float>::epsilon();
  double constexpr dTol = std::numeric_limits<double>::epsilon();
  // Test casting Matrix from float to double (compile-time size)
  {
    Matrix<float, 3, 4> srcFloat;
    srcFloat.SetRandom(42);

    auto castDouble = StaticCast<Matrix<double, 3, 4>>(srcFloat);

    EXPECT_EQ(srcFloat.Rows(), castDouble.Rows());
    EXPECT_EQ(srcFloat.Cols(), castDouble.Cols());
    EXPECT_TRUE(NearEqualMatrices(srcFloat, castDouble, fTol));

    // Verify type conversion
    static_assert(
        std::is_same_v<typename decltype(castDouble)::Scalar, double>,
        "Expected double scalar type");
  }

  // Test casting Matrix from double to float (dynamic size)
  {
    Matrix<double> srcDouble(5, 6);
    srcDouble.SetRandom(123);

    auto castFloat = StaticCast<Matrix<float>>(srcDouble);

    EXPECT_EQ(srcDouble.Rows(), castFloat.Rows());
    EXPECT_EQ(srcDouble.Cols(), castFloat.Cols());
    EXPECT_TRUE(NearEqualMatrices(castFloat, srcDouble, fTol));
    // Check that some data is lost in the conversion
    EXPECT_FALSE(NearEqualMatrices(srcDouble, castFloat, dTol));

    // Verify type conversion
    static_assert(
        std::is_same_v<typename decltype(castFloat)::Scalar, float>, "Expected float scalar type");
  }

  // Test casting sub-block Matrix from double to float
  {
    Matrix<double> initDouble(8, 7);
    initDouble.SetRandom(123);
    auto srcDouble = initDouble.Block(2, 3, 3, 2);

    auto castFloat = StaticCast<Matrix<float>>(srcDouble);

    EXPECT_EQ(srcDouble.Rows(), castFloat.Rows());
    EXPECT_EQ(srcDouble.Cols(), castFloat.Cols());
    EXPECT_TRUE(NearEqualMatrices(castFloat, srcDouble, fTol));
    // Check that some data is lost in the conversion
    EXPECT_FALSE(NearEqualMatrices(srcDouble, castFloat, dTol));

    // Verify type conversion
    static_assert(
        std::is_same_v<typename decltype(castFloat)::Scalar, float>, "Expected float scalar type");
  }

  // Test with RowMajor matrix
  {
    RowMatrix<float, 4, 5> srcRowMajor;
    srcRowMajor.SetRandom(99);

    auto castDouble = StaticCast<RowMatrix<double, 4, 5>>(srcRowMajor);

    EXPECT_EQ(srcRowMajor.Rows(), castDouble.Rows());
    EXPECT_EQ(srcRowMajor.Cols(), castDouble.Cols());
    EXPECT_TRUE(NearEqualMatrices(srcRowMajor, castDouble, fTol));
  }

  // Test with empty matrix
  {
    Matrix<float> srcEmpty;
    auto castEmpty = StaticCast<Matrix<double>>(srcEmpty);
    EXPECT_EQ(0, castEmpty.Rows());
    EXPECT_EQ(0, castEmpty.Cols());
  }

  // Test with vectors
  {
    ColumnVector<float, 7> srcVector;
    srcVector.SetRandom(55);

    auto castDouble = StaticCast<ColumnVector<double, 7>>(srcVector);

    EXPECT_EQ(srcVector.Rows(), castDouble.Rows());
    EXPECT_EQ(srcVector.Cols(), castDouble.Cols());
    EXPECT_TRUE(NearEqualMatrices(srcVector, castDouble, fTol));
  }
}

TEST(StaticCast, SparseMatrix) {
  float constexpr fTol = std::numeric_limits<float>::epsilon();
  double constexpr dTol = std::numeric_limits<double>::epsilon();
  // Test casting SparseMatrix from float to double
  {
    // |1 2 0|
    // |0 3 4|
    // |5 0 6|
    SparseMatrix<float> srcFloat(
        3,
        DynamicArray<int>{0, 2, 4, 6},
        DynamicArray<int>{0, 1, 1, 2, 0, 2},
        DynamicArray<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    auto castDouble = StaticCast<SparseMatrix<double>>(srcFloat);

    EXPECT_EQ(srcFloat.Rows(), castDouble.Rows());
    EXPECT_EQ(srcFloat.Cols(), castDouble.Cols());
    EXPECT_EQ(srcFloat.NumNonZeros(), castDouble.NumNonZeros());
    EXPECT_TRUE(NearEqualSpan(srcFloat.Pointers(), castDouble.Pointers()));
    EXPECT_TRUE(NearEqualSpan(srcFloat.Indices(), castDouble.Indices()));
    EXPECT_TRUE(NearEqualSpan(srcFloat.Values(), castDouble.Values(), fTol));

    // Verify type conversion
    static_assert(
        std::is_same_v<typename decltype(castDouble)::Scalar, double>,
        "Expected double scalar type");
  }

  // Test casting SparseMatrix from double to float
  {
    SparseMatrix<double> srcDouble(
        2,
        DynamicArray<int>{0, 1, 3},
        DynamicArray<int>{1, 0, 1},
        DynamicArray<double>{-0.1234567890123, 0.35, 0.123456789876});

    auto castFloat = StaticCast<SparseMatrix<float>>(srcDouble);

    EXPECT_EQ(srcDouble.Rows(), castFloat.Rows());
    EXPECT_EQ(srcDouble.Cols(), castFloat.Cols());
    EXPECT_EQ(srcDouble.NumNonZeros(), castFloat.NumNonZeros());
    EXPECT_TRUE(NearEqualSpan(castFloat.Pointers(), srcDouble.Pointers()));
    EXPECT_TRUE(NearEqualSpan(castFloat.Indices(), srcDouble.Indices()));
    EXPECT_TRUE(NearEqualSpan(castFloat.Values(), srcDouble.Values(), fTol));
    // Check that some data is lost in the conversion
    EXPECT_FALSE(NearEqualSpan(srcDouble.Values(), castFloat.Values(), dTol));

    // Verify type conversion
    static_assert(
        std::is_same_v<typename decltype(castFloat)::Scalar, float>, "Expected float scalar type");
  }

  // Test with empty matrix
  {
    SparseMatrix<float> srcEmpty;
    auto castEmpty = StaticCast<SparseMatrix<double>>(srcEmpty);
    EXPECT_EQ(0, castEmpty.Rows());
    EXPECT_EQ(0, castEmpty.Cols());
    EXPECT_EQ(0, castEmpty.NumNonZeros());
  }
}

TEST(StaticCast, AnyMatrix) {
  float constexpr fTol = std::numeric_limits<float>::epsilon();
  double constexpr dTol = std::numeric_limits<double>::epsilon();
  int constexpr kMatIdx = 3;
  int constexpr kSpMatIdx = 2;
  int constexpr kBlkSpMatIdx = 0;
  // Test casting AnyMatrix containing a Matrix
  {
    Matrix<float, 3, 3> matFloat;
    matFloat.SetRandom(42);
    AnyMatrix<float> anyMatFloat{matFloat};

    auto anyMatDouble = StaticCast<AnyMatrix<double>>(anyMatFloat);
    EXPECT_EQ(anyMatDouble.index(), anyMatFloat.index());

    EXPECT_TRUE(
        NearEqualMatrices(std::get<kMatIdx>(anyMatFloat), std::get<kMatIdx>(anyMatDouble), fTol));
  }

  // Test casting AnyMatrix containing a SparseMatrix
  {
    SparseMatrix<double> spMatDouble(
        2,
        DynamicArray<int>{0, 1, 2},
        DynamicArray<int>{0, 1},
        DynamicArray<double>{1.56789012345, 2.543210987654});
    AnyMatrix<double> anyMatDouble{spMatDouble};

    auto anyMatFloat = StaticCast<AnyMatrix<float>>(anyMatDouble);
    EXPECT_EQ(anyMatDouble.index(), anyMatFloat.index());
    auto spMatFloat = std::get<kSpMatIdx>(anyMatFloat);
    EXPECT_EQ(spMatFloat.Rows(), spMatDouble.Rows());
    EXPECT_EQ(spMatFloat.Cols(), spMatDouble.Cols());
    EXPECT_TRUE(NearEqualSpan(spMatDouble.Pointers(), spMatFloat.Pointers()));
    EXPECT_TRUE(NearEqualSpan(spMatDouble.Indices(), spMatFloat.Indices()));

    EXPECT_TRUE(NearEqualSpan(spMatFloat.Values(), spMatDouble.Values(), fTol));
    // Check that some data is lost in the conversion
    EXPECT_FALSE(NearEqualSpan(spMatDouble.Values(), spMatFloat.Values(), dTol));
  }

  // Test casting AnyMatrix containing a BlockSparseMatrix
  {
    BlockSparseMatrix<float, 3> blkMatFloat{
        2,
        DynamicArray<int>{0, 1},
        DynamicArray<int>{0},
        DynamicArray<float>{1.2345678f, 2.1313131313f, 3.0f, 4.0f, 5.0f, 6.1f, 7.2f, 8.3f, 9.4f}};
    AnyMatrix<float> anyMatFloat{blkMatFloat};

    auto anyMatDouble = StaticCast<AnyMatrix<double>>(anyMatFloat);
    EXPECT_EQ(anyMatDouble.index(), anyMatFloat.index());
    auto blkMatDouble = std::get<kBlkSpMatIdx>(anyMatDouble);
    EXPECT_EQ(blkMatFloat.Rows(), blkMatDouble.Rows());
    EXPECT_EQ(blkMatFloat.Cols(), blkMatDouble.Cols());
    EXPECT_TRUE(NearEqualSpan(blkMatDouble.Pointers(), blkMatFloat.Pointers()));
    EXPECT_TRUE(NearEqualSpan(blkMatDouble.Indices(), blkMatFloat.Indices()));

    EXPECT_TRUE(NearEqualSpan(blkMatFloat.Values(), blkMatDouble.Values(), fTol));
  }
}

TEST(StaticCast, AnyMatrixView) {
  float constexpr fTol = std::numeric_limits<float>::epsilon();
  double constexpr dTol = std::numeric_limits<double>::epsilon();
  int constexpr kMatIdx = 3;
  int constexpr kSpMatIdx = 2;
  // Test casting AnyMatrixView from float to double
  {
    Matrix<float, 4, 4> matFloat;
    matFloat.SetRandom(99);
    AnyMatrixView<float const> viewFloat = AsConstView(matFloat);

    auto anyMatDouble = StaticCast<AnyMatrix<double>>(viewFloat);
    EXPECT_EQ(anyMatDouble.index(), viewFloat.index());

    EXPECT_TRUE(
        NearEqualMatrices(std::get<kMatIdx>(viewFloat), std::get<kMatIdx>(anyMatDouble), fTol));
  }

  // Test casting AnyMatrixView from double to float
  {
    SparseMatrix<double> spMatDouble(
        3,
        DynamicArray<int>{0, 1, 2, 3},
        DynamicArray<int>{1, 0, 2},
        DynamicArray<double>{1.1, 2.2, 3.3});
    AnyMatrixView<double const> viewDouble = AsConstView(spMatDouble);

    auto anyMatFloat = StaticCast<AnyMatrix<float>>(viewDouble);
    EXPECT_EQ(anyMatFloat.index(), viewDouble.index());

    auto spMatFloat = std::get<kSpMatIdx>(anyMatFloat);
    EXPECT_EQ(spMatFloat.Rows(), spMatDouble.Rows());
    EXPECT_EQ(spMatFloat.Cols(), spMatDouble.Cols());
    EXPECT_TRUE(NearEqualSpan(spMatDouble.Pointers(), spMatFloat.Pointers()));
    EXPECT_TRUE(NearEqualSpan(spMatDouble.Indices(), spMatFloat.Indices()));

    EXPECT_TRUE(NearEqualSpan(spMatFloat.Values(), spMatDouble.Values(), fTol));
    // Check that some data is lost in the conversion
    EXPECT_FALSE(NearEqualSpan(spMatDouble.Values(), spMatFloat.Values(), dTol));
  }
}

TEST(StaticCast, IslandOperators) {
  float constexpr fTol = std::numeric_limits<float>::epsilon();
  double constexpr dTol = std::numeric_limits<double>::epsilon();
  int constexpr kMatIdx = 3;
  int constexpr kSpMatIdx = 2;
  // Create a simple IslandOperators with actor and interaction matrices
  {
    // Create actor matrices
    std::vector<std::pair<int, AnyMatrix<float>>> actorMatrices;
    Matrix<float, 3, 3> actorMat1;
    actorMat1.SetRandom(11);
    actorMatrices.push_back({0, AnyMatrix<float>{actorMat1}});

    Matrix<float, 2, 2> actorMat2;
    actorMat2.SetRandom(22);
    actorMatrices.push_back({3, AnyMatrix<float>{actorMat2}});

    // Create interaction matrices
    std::vector<AnyInteractionMatrixInfo<float>> interactionMatrices;
    SparseMatrix<float> interMat1(
        2, DynamicArray<int>{0, 1, 2}, DynamicArray<int>{0, 1}, DynamicArray<float>{1.0f, 2.0f});
    interactionMatrices.emplace_back(0, 3, AnyMatrix<float>{interMat1}, std::nullopt);

    // Create IslandOperators
    IslandOperatorsOwningLite<float> opsOwningFloat{
        std::move(actorMatrices), std::move(interactionMatrices)};
    IslandOperators<float> opsFloat = opsOwningFloat.AsConstView();

    // Cast to double
    auto opsOwningDouble = StaticCast<IslandOperatorsOwningLite<double>>(opsFloat);
    auto opsDouble = opsOwningDouble.AsConstView();

    // Verify actor matrices
    EXPECT_EQ(opsFloat.GetActorMatrices().size(), opsDouble.GetActorMatrices().size());
    for (size_t i = 0; i < opsFloat.GetActorMatrices().size(); ++i) {
      auto const& srcActor = opsFloat.GetActorMatrices()[i];
      auto const& dstActor = opsOwningDouble.actorMatrices[i];
      EXPECT_EQ(std::get<0>(srcActor), std::get<0>(dstActor));
      EXPECT_TRUE(NearEqualMatrices(
          std::get<kMatIdx>(std::get<1>(srcActor)),
          std::get<kMatIdx>(std::get<1>(dstActor)),
          fTol));
    }

    // Verify interaction matrices
    EXPECT_EQ(opsFloat.GetInteractionMatrices().size(), opsDouble.GetInteractionMatrices().size());
    for (size_t i = 0; i < opsFloat.GetInteractionMatrices().size(); ++i) {
      auto const& srcInter = opsFloat.GetInteractionMatrices()[i];
      auto const& dstInter = opsDouble.GetInteractionMatrices()[i];
      EXPECT_EQ(srcInter.rowOffset, dstInter.rowOffset);
      EXPECT_EQ(srcInter.colOffset, dstInter.colOffset);
      auto const& spMatFloat = std::get<kSpMatIdx>(srcInter.matrix);
      auto const& spMatDouble = std::get<kSpMatIdx>(dstInter.matrix);
      EXPECT_TRUE(NearEqualSpan(spMatFloat.Pointers(), spMatDouble.Pointers()));
      EXPECT_TRUE(NearEqualSpan(spMatFloat.Indices(), spMatDouble.Indices()));
      EXPECT_TRUE(NearEqualSpan(spMatFloat.Values(), spMatDouble.Values(), fTol));
    }

    // Verify type conversion
    static_assert(
        std::is_same_v<typename decltype(opsDouble)::Scalar, double>,
        "Expected double scalar type");
  }

  // Test with empty IslandOperators
  {
    IslandOperators<float> opsEmpty({}, {}, {});
    auto opsOwningDouble = StaticCast<IslandOperatorsOwningLite<double>>(opsEmpty);
    auto opsDouble = opsOwningDouble.AsConstView();

    EXPECT_EQ(0, opsDouble.GetActorMatrices().size());
    EXPECT_EQ(0, opsDouble.GetInteractionMatrices().size());
  }

  // Test double to float conversion
  {
    std::vector<std::pair<int, AnyMatrix<double>>> actorMatrices;
    Matrix<double, 2, 2> actorMat;
    actorMat.SetRandom(55);
    actorMatrices.push_back({0, AnyMatrix<double>{actorMat}});

    IslandOperatorsOwningLite<double> opsOwningDouble{std::move(actorMatrices), {}};
    IslandOperators<double> opsDouble = opsOwningDouble.AsConstView();

    auto opsOwningFloat = StaticCast<IslandOperatorsOwningLite<float>>(opsDouble);
    auto opsFloat = opsOwningFloat.AsConstView();

    EXPECT_EQ(opsDouble.GetActorMatrices().size(), opsFloat.GetActorMatrices().size());
    EXPECT_TRUE(NearEqualMatrices(
        std::get<kMatIdx>(std::get<1>(opsFloat.GetActorMatrices()[0])),
        std::get<kMatIdx>(std::get<1>(opsDouble.GetActorMatrices()[0])),
        fTol));
    EXPECT_FALSE(NearEqualMatrices(
        std::get<kMatIdx>(std::get<1>(opsDouble.GetActorMatrices()[0])),
        std::get<kMatIdx>(std::get<1>(opsFloat.GetActorMatrices()[0])),
        dTol));

    // Verify type conversion
    static_assert(
        std::is_same_v<typename decltype(opsFloat)::Scalar, float>, "Expected float scalar type");
  }
}

TEST(MatrixConversion, AsBlockSparseMatrixConstView) {
  // Create a dense matrix with random values and convert to sparse
  RowMatrix<real> dense(20, 25);
  dense.SetRandom(42);
  auto sparse = ToSparseMatrix(dense);

  // Convert to block sparse view
  auto blockView = AsBlockSparseMatrixConstView(sparse);

  // Verify dimensions match
  EXPECT_EQ(blockView.Rows(), sparse.Rows());
  EXPECT_EQ(blockView.Cols(), sparse.Cols());
  EXPECT_EQ(blockView.NumNonZeros(), sparse.NumNonZeros());

  // Verify block size is 1
  static_assert(std::remove_cvref_t<decltype(blockView)>::kBlockSize == 1);

  // Verify data is equivalent
  EXPECT_EQ(blockView.Values().data(), sparse.Values().data());
  EXPECT_TRUE(NearEqualMatrices(sparse, blockView));
}
