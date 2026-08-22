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

#include <mochi_core/linear_algebra/cuda/cuda_matrix.h>
#include <mochi_core/linear_algebra/matrix.h>

#include <gtest/gtest.h>

using namespace mochi;

TEST(MatTraits, kRows) {
  {
    using T = Matrix<float>;
    EXPECT_EQ(krylov::details::MatTraits<T>::kNumRows, krylov::kDynamic);
  }
  {
    using T = Matrix<float, 5, 7>;
    EXPECT_EQ(krylov::details::MatTraits<T>::kNumRows, 5);
  }
}

TEST(MatTraits, kCols) {
  {
    using T = Matrix<float>;
    EXPECT_EQ(krylov::details::MatTraits<T>::kNumCols, krylov::kDynamic);
  }
  {
    using T = Matrix<float, 5, 7>;
    EXPECT_EQ(krylov::details::MatTraits<T>::kNumCols, 7);
  }
}

TEST(MatTraits, kMajorDir) {
  {
    using T = Matrix<float>;
    EXPECT_EQ(krylov::details::MatTraits<T>::kMajorDir, krylov::Direction::ColMajor);
    using S = Matrix<float, krylov::kDynamic, krylov::kDynamic, krylov::Direction::RowMajor>;
    EXPECT_EQ(krylov::details::MatTraits<S>::kMajorDir, krylov::Direction::RowMajor);
  }
  {
    using T = Matrix<float, 5, 7>;
    EXPECT_EQ(krylov::details::MatTraits<T>::kMajorDir, krylov::Direction::ColMajor);
    using S = Matrix<float, 3, 5, krylov::Direction::RowMajor>;
    EXPECT_EQ(krylov::details::MatTraits<S>::kMajorDir, krylov::Direction::RowMajor);
  }
}

TEST(MatTraits, kLeadDim) {
  {
    using S = Matrix<
        float,
        3,
        5,
        krylov::Direction::RowMajor,
        krylov::Ownership::Owner,
        krylov::kDynamic>;
    EXPECT_EQ(krylov::details::MatTraits<S>::kLeadDim, krylov::kDynamic);
    using T = Matrix<float>;
    EXPECT_EQ(krylov::details::MatTraits<T>::kLeadDim, krylov::kAutomaticLeadDim);
    using U = Matrix<float, 3, 5, krylov::Direction::RowMajor, krylov::Ownership::Owner, 7>;
    EXPECT_EQ(krylov::details::MatTraits<U>::kLeadDim, 7);
  }
}

TEST(MatTraits, kCuda) {
  {
    using T = Matrix<float>;
    EXPECT_FALSE(krylov::details::MatTraits<T>::kIsCuda);
  }
  {
    using T = CudaMatrix<float>;
    EXPECT_TRUE(krylov::details::MatTraits<T>::kIsCuda);
  }
  {
    using T = CudaVector<float>;
    EXPECT_TRUE(krylov::details::MatTraits<T>::kIsCuda);
  }
}
