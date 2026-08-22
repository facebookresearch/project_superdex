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

#include <mochi_core/linear_algebra/krylov/tools/tensor_functions.h>
#include <mochi_core/linear_algebra/matrix.h>

#include "matrix_expression_test.h"

namespace mochi::test {

template <
    typename Scalar,
    int mAtCT,
    int nAtCT,
    int kAtCT,
    krylov::Direction kDir,
    krylov::Ownership kOwner>
void TestDirectProductLevel2(int m, int n, int k) {
  // Level 2 wrapper for 'DirectProduct' tests.
  EXPECT_GT(m, 0);
  EXPECT_GT(n, 0);
  EXPECT_GT(k, 0);

  mochi::Matrix<Scalar, mAtCT, kAtCT, kDir, kOwner> A(m, k);
  mochi::Matrix<Scalar, kAtCT, nAtCT, kDir, kOwner> B1(k, n);
  mochi::Matrix<Scalar, kAtCT, nAtCT, ~kDir, kOwner> B2(k, n);
  mochi::Matrix<Scalar, mAtCT, nAtCT, kDir, kOwner> C1(m, n);
  mochi::Matrix<Scalar, mAtCT, nAtCT, ~kDir, kOwner> C2(m, n);
  mochi::Matrix<Scalar, mAtCT, nAtCT> D(m, n);

  auto const scale = Scalar(3.14);
  A.SetRandom(static_cast<unsigned int>(m + n + k));
  B1.SetRandom(static_cast<unsigned int>(2 + m + n + k));
  B2.SetRandom(static_cast<unsigned int>(10 + m + n + k));

  C1 = A * B1;
  MultiplyMatrices(Scalar(1), A, B1, D);
  EXPECT_TRUE(IsNear(C1, D, GetTol<Scalar>(k)));

  C1 = scale * A * B2;
  MultiplyMatrices(scale, A, B2, D);
  EXPECT_TRUE(IsNear(C1, D, scale * GetTol<Scalar>(k)));

  C2 = scale * A * B1;
  MultiplyMatrices(scale, A, B1, D);
  EXPECT_TRUE(IsNear(C2, D, scale * GetTol<Scalar>(k)));

  C2 = A * B2;
  MultiplyMatrices(Scalar(1), A, B2, D);
  EXPECT_TRUE(IsNear(C2, D, GetTol<Scalar>(k)));

  // Using 'Apply'.
  C1.SetRandom(123);
  MultiplyMatrices(Scalar(1), A, B1, D);
  EXPECT_FALSE(IsNear(C1, D, GetTol<Scalar>(k)));
  Apply(A, B1, C1);
  EXPECT_TRUE(IsNear(C1, D, GetTol<Scalar>(k)));

  C1.SetRandom(234);
  MultiplyMatrices(Scalar(1), A, B2, D);
  EXPECT_FALSE(IsNear(C1, D, GetTol<Scalar>(k)));
  Apply(A, B2, C1);
  EXPECT_TRUE(IsNear(C1, D, GetTol<Scalar>(k)));

  C2.SetRandom(345);
  MultiplyMatrices(Scalar(1), A, B1, D);
  EXPECT_FALSE(IsNear(C2, D, GetTol<Scalar>(k)));
  Apply(A, B1, C2);
  EXPECT_TRUE(IsNear(C2, D, GetTol<Scalar>(k)));

  C2.SetRandom(456);
  MultiplyMatrices(Scalar(1), A, B2, D);
  EXPECT_FALSE(IsNear(C2, D, GetTol<Scalar>(k)));
  Apply(A, B2, C2);
  EXPECT_TRUE(IsNear(C2, D, GetTol<Scalar>(k)));
}

template <
    typename Scalar,
    krylov::Direction kDir,
    krylov::Ownership kOwner = krylov::Ownership::Owner>
void TestDirectProductLevel1() {
  // Level 1 wrapper for 'DirectProduct' tests.

  // Various compile-time m
  TestDirectProductLevel2<Scalar, 2, 3, 3, kDir, kOwner>(2, 3, 3);
  TestDirectProductLevel2<Scalar, 3, 3, 3, kDir, kOwner>(3, 3, 3);
  TestDirectProductLevel2<Scalar, 4, 3, 3, kDir, kOwner>(4, 3, 3);
  TestDirectProductLevel2<Scalar, 7, 3, 3, kDir, kOwner>(7, 3, 3);
  TestDirectProductLevel2<Scalar, 8, 3, 3, kDir, kOwner>(8, 3, 3);
  TestDirectProductLevel2<Scalar, 9, 3, 3, kDir, kOwner>(9, 3, 3);

  // Various compile-time n
  TestDirectProductLevel2<Scalar, 3, 2, 3, kDir, kOwner>(3, 2, 3);
  TestDirectProductLevel2<Scalar, 3, 3, 3, kDir, kOwner>(3, 3, 3);
  TestDirectProductLevel2<Scalar, 3, 4, 3, kDir, kOwner>(3, 4, 3);
  TestDirectProductLevel2<Scalar, 3, 7, 3, kDir, kOwner>(3, 7, 3);
  TestDirectProductLevel2<Scalar, 3, 8, 3, kDir, kOwner>(3, 8, 3);
  TestDirectProductLevel2<Scalar, 3, 9, 3, kDir, kOwner>(3, 9, 3);

  // Various compile-time k
  TestDirectProductLevel2<Scalar, 3, 3, 2, kDir, kOwner>(3, 3, 2);
  TestDirectProductLevel2<Scalar, 3, 3, 3, kDir, kOwner>(3, 3, 3);
  TestDirectProductLevel2<Scalar, 3, 3, 4, kDir, kOwner>(3, 3, 4);
  TestDirectProductLevel2<Scalar, 3, 3, 7, kDir, kOwner>(3, 3, 7);
  TestDirectProductLevel2<Scalar, 3, 3, 8, kDir, kOwner>(3, 3, 8);
  TestDirectProductLevel2<Scalar, 3, 3, 9, kDir, kOwner>(3, 3, 9);

  // Dynamic dimensions
  constexpr int kSizes[] = {
      1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 31, 32, 33,
#if MOCHI_OPTIMIZED // Expensive cases. Only in optimized builds.
      18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 63, 64, 65
#endif
  };
  for (int m : kSizes) {
    for (int n : kSizes) {
      for (int k : kSizes) {
        TestDirectProductLevel2<
            Scalar,
            krylov::kDynamic,
            krylov::kDynamic,
            krylov::kDynamic,
            kDir,
            kOwner>(m, n, k);
      }
    }
  }
}

} // namespace mochi::test
