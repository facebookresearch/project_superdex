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

#include <mochi_core/linear_algebra/block_one_d_view.h>
#include <mochi_core/linear_algebra/block_view_vector.h>
#include <mochi_core/test/mochi_test_helpers.h>

#include <vector>

using namespace mochi;

/**************************************************************************************
 * BlockRowView Examples
 */

TEST(BlockRowView, Dimensions) {
  constexpr int nRow = 3;
  int nCol = 9; // nCol must be a multiple of nrow
  std::vector<real> values(nRow * nCol);
  for (int i = 0; i < nRow; ++i) {
    for (int j = 0; j < nCol; ++j) {
      values[j + i * nCol] = real(i + j + 1);
    }
  }
  BlockRowView<real, nRow> R(values.data(), nCol, nCol / nRow);
  EXPECT_EQ(nCol / nRow, R.NumBlocks());
  EXPECT_EQ(nCol, R.LeadDim());
  //--- Test public members
  EXPECT_EQ(nRow, R.Underlying().Rows());
  EXPECT_EQ(nCol / nRow, R.numBlocks);
}

template <krylov::Direction kDir>
void BlockRowColView_Data_Test() {
  constexpr int kBlockCount = 3;
  constexpr int kBlockSize = 3;
  constexpr int kLeadDim = kBlockCount * kBlockSize;
  constexpr bool kByRow = kDir == krylov::Direction::RowMajor;
  constexpr int nRow = kByRow ? kBlockSize : kLeadDim;
  constexpr int nCol = kByRow ? kLeadDim : kBlockSize;
  constexpr int kColStride = kByRow ? 1 : kLeadDim;
  constexpr int kRowStride = kByRow ? kLeadDim : 1;
  std::vector<real> values(nRow * nCol);
  for (int r = 0; r < nRow; ++r) {
    for (int c = 0; c < nCol; ++c) {
      values[c * kColStride + r * kRowStride] = static_cast<real>(r + c + 1);
    }
  }
  BlockOneDView<real, kBlockSize, kDir> R(values.data(), kLeadDim, kBlockCount);
  EXPECT_EQ(values.data(), &R(0, 0)); // Check BlockRowView creates a view
  EXPECT_EQ(&R(0, 0), R.Data());
  for (int r = 0; r < nRow; ++r) {
    for (int c = 0; c < nCol; ++c) {
      EXPECT_EQ(values[c * kColStride + r * kRowStride], R(r, c));
    }
  }
  for (int block = 0; block < kBlockCount; ++block) {
    auto B = R[block];
    for (int r = 0; r < kBlockSize; ++r) {
      for (int c = 0; c < kBlockSize; ++c) {
        int fullC = kByRow ? c + block * kBlockSize : c;
        int fullR = kByRow ? r : r + block * kBlockSize;
        EXPECT_EQ(static_cast<real>(fullR + fullC + 1), B(r, c));
      }
    }
  }
  R.SetZero();
  //
  for (int i = 0; i < nRow * nCol; ++i) {
    EXPECT_EQ(real(0), values[i]);
  }
}

TEST(BlockRowColView, Values) {
  BlockRowColView_Data_Test<krylov::Direction::ColMajor>();
  BlockRowColView_Data_Test<krylov::Direction::RowMajor>();
}

TEST(BlockRowView, Values) {
  constexpr int nRow = 3;
  int nCol = 9; // nCol must be a multiple of nrow
  std::vector<real> values(nRow * nCol);
  for (int i = 0; i < nRow; ++i) {
    for (int j = 0; j < nCol; ++j) {
      values[j + i * nCol] = real(i + j + 1);
    }
  }
  BlockRowView<real, nRow> R(values.data(), nCol, nCol / nRow);
  EXPECT_EQ(values.data(), &R(0, 0)); // Check BlockRowView creates a view
  EXPECT_EQ(&R(0, 0), R.Data());
  EXPECT_EQ(&R(0, 0), R.data());
  for (int i = 0; i < nRow; ++i) {
    for (int j = 0; j < nCol; ++j) {
      EXPECT_EQ(real(i + j + 1), R(i, j));
    }
  }
  //
  {
    auto b0 = R[0];
    EXPECT_EQ(&R(0, 0), &b0(0, 0)); // Check operator[] creates a view
    EXPECT_EQ(&b0(0, 0), b0.Data());
    EXPECT_EQ(&b0(0, 0), b0.data());
    EXPECT_EQ(real(1), b0(0, 0));
    EXPECT_EQ(real(2), b0(0, 1));
    EXPECT_EQ(real(3), b0(0, 2));
    EXPECT_EQ(real(2), b0(1, 0));
    EXPECT_EQ(real(3), b0(1, 1));
    EXPECT_EQ(real(4), b0(1, 2));
    EXPECT_EQ(real(3), b0(2, 0));
    EXPECT_EQ(real(4), b0(2, 1));
    EXPECT_EQ(real(5), b0(2, 2));
  }
  //
  {
    auto b1 = R[1];
    EXPECT_EQ(&b1(0, 0), b1.Data());
    EXPECT_EQ(&b1(0, 0), b1.data());
    EXPECT_EQ(real(4), b1(0, 0));
    EXPECT_EQ(real(5), b1(0, 1));
    EXPECT_EQ(real(6), b1(0, 2));
    EXPECT_EQ(real(5), b1(1, 0));
    EXPECT_EQ(real(6), b1(1, 1));
    EXPECT_EQ(real(7), b1(1, 2));
    EXPECT_EQ(real(6), b1(2, 0));
    EXPECT_EQ(real(7), b1(2, 1));
    EXPECT_EQ(real(8), b1(2, 2));
  }
  //
  {
    auto b2 = R[2];
    EXPECT_EQ(real(7), b2(0, 0));
    EXPECT_EQ(real(8), b2(0, 1));
    EXPECT_EQ(real(9), b2(0, 2));
    EXPECT_EQ(real(8), b2(1, 0));
    EXPECT_EQ(real(9), b2(1, 1));
    EXPECT_EQ(real(10), b2(1, 2));
    EXPECT_EQ(real(9), b2(2, 0));
    EXPECT_EQ(real(10), b2(2, 1));
    EXPECT_EQ(real(11), b2(2, 2));
  }
  //
  R.SetZero();
  //
  for (int i = 0; i < nRow * nCol; ++i) {
    EXPECT_EQ(real(0), values[i]);
  }
}

TEST(BlockRowView, FirstBlocks) {
  constexpr int nRow = 3;
  int nCol = 9; // nCol must be a multiple of nrow
  std::vector<real> values(nRow * nCol);
  for (int i = 0; i < nRow; ++i) {
    for (int j = 0; j < nCol; ++j) {
      values[j + i * nCol] = real(i + j + 1);
    }
  }
  BlockRowView<real, nRow> R(values.data(), nCol, nCol / nRow);
  //
  {
    auto R0 = R.FirstBlocks(0);
    EXPECT_EQ(0, R0.Underlying().Cols());
  }
  //
  {
    auto R1 = R.FirstBlocks(1);
    EXPECT_EQ(3, R1.Underlying().Cols());
    EXPECT_EQ(&R(0, 0), &R1(0, 0)); // Check FirstBlocks creates a view
    EXPECT_EQ(&R1(0, 0), R1.Data());
    EXPECT_EQ(&R1(0, 0), R1.data());
    for (int i = 0; i < nRow; ++i) {
      for (int j = 0; j < nRow; ++j) {
        EXPECT_EQ(real(i + j + 1), R1(i, j));
      }
    }
  }
  //
  {
    auto R2 = R.FirstBlocks(2);
    EXPECT_EQ(6, R2.Underlying().Cols());
    for (int i = 0; i < nRow; ++i) {
      for (int j = 0; j < 2 * nRow; ++j) {
        EXPECT_EQ(real(i + j + 1), R2(i, j));
      }
    }
  }
  //
  {
    auto R3 = R.FirstBlocks(3);
    EXPECT_EQ(9, R3.Underlying().Cols());
  }
}

TEST(BlockRowView, LastBlocks) {
  constexpr int nRow = 3;
  int nCol = 9; // nCol must be a multiple of nrow
  std::vector<real> values(nRow * nCol);
  for (int i = 0; i < nRow; ++i) {
    for (int j = 0; j < nCol; ++j) {
      values[j + i * nCol] = real(i + j + 1);
    }
  }
  BlockRowView<real, nRow> R(values.data(), nCol, nCol / nRow);
  //
  {
    auto R0 = R.LastBlocks(0);
    EXPECT_EQ(0, R0.Underlying().Cols());
  }
  //
  {
    auto R1 = R.LastBlocks(1);
    EXPECT_EQ(3, R1.Underlying().Cols());
    EXPECT_EQ(&R(0, 6), &R1(0, 0)); // Check LastBlocks creates a view
    EXPECT_EQ(&R1(0, 0), R1.Data());
    EXPECT_EQ(&R1(0, 0), R1.data());
    for (int i = 0; i < nRow; ++i) {
      for (int j = 0; j < nRow; ++j) {
        EXPECT_EQ(real(i + j + 7), R1(i, j));
      }
    }
  }
  //
  {
    auto R2 = R.LastBlocks(2);
    EXPECT_EQ(6, R2.Underlying().Cols());
    for (int i = 0; i < nRow; ++i) {
      for (int j = 0; j < 2 * nRow; ++j) {
        EXPECT_EQ(real(i + j + 4), R2(i, j));
      }
    }
  }
  //
  {
    auto R3 = R.LastBlocks(3);
    EXPECT_EQ(9, R3.Underlying().Cols());
  }
}

/**************************************************************************************
 * BlockViewVector Examples
 */

TEST(BlockViewVector, Dimension) {
  constexpr int kBlockSize = 3;
  int nRows = kBlockSize * 4;
  std::vector<real> values(nRows);
  for (int i = 0; i < nRows; ++i) {
    values[i] = i + 1;
  }
  BlockViewVector<real, kBlockSize> R(values.data(), nRows / kBlockSize);
  EXPECT_EQ(nRows / kBlockSize, R.BlockRows());
}

TEST(BlockViewVector, Values) {
  constexpr int kBlockSize = 3;
  int nRows = kBlockSize * 4;
  std::vector<real> values(nRows);
  for (int i = 0; i < nRows; ++i) {
    values[i] = i + 1;
  }
  BlockViewVector<real, kBlockSize> R(values.data(), nRows / kBlockSize);
  EXPECT_EQ(values.data(), R.Data()); // Check BlockViewVector creates a view
  {
    auto b0 = R[0];
    EXPECT_EQ(R.Data(), &b0(0, 0)); // Check operator[] creates a view
    EXPECT_EQ(real(1), b0(0, 0));
    EXPECT_EQ(real(2), b0(1, 0));
    EXPECT_EQ(real(3), b0(2, 0));
  }
  {
    auto b1 = R[1];
    EXPECT_EQ(real(4), b1(0, 0));
    EXPECT_EQ(real(5), b1(1, 0));
    EXPECT_EQ(real(6), b1(2, 0));
  }
  {
    auto b2 = R[2];
    EXPECT_EQ(real(7), b2(0, 0));
    EXPECT_EQ(real(8), b2(1, 0));
    EXPECT_EQ(real(9), b2(2, 0));
  }
  //
  R.SetZero();
  //
  for (int i = 0; i < nRows; ++i) {
    EXPECT_EQ(real(0), values[i]);
  }
}
