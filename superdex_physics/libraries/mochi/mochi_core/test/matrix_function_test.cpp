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

/**
 * @file matrix_function_test.cpp
 * @brief Tests Matrix class (matrix.h) behaviors that require targeted, non-parametric
 * verification.
 *
 * @details Covers move semantics, swap, constructors with allocators, expression construction, view
 * Reset(), operator bool(), initializer lists, factory methods (Zero, Ones, Random), SetIdentity
 * (including on views), SetConstant, memory layout guarantees, etc.
 *
 * @note For parametric testing of element access, views, blocks, norms, and arithmetic across all
 * size/direction/scalar combinations, see matrix_test.h.
 */

#include <mochi_core/linear_algebra/krylov/gmres.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/test/allocator_test_helpers.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/math_utils.h>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

// TODO[T217402682]
#if MOCHI_COMPILER_GCC
MOCHI_WARNING_IGNORE_GCC_CLANG(GCC diagnostic ignored "-Wstringop-overread")
#endif

using namespace mochi;
using namespace mochi::test;

template <typename MA, typename MB>
static bool AreSame(MA const& A, MB const& B) {
  if (A.Rows() != B.Rows() || A.Cols() != B.Cols()) {
    return false;
  }
  for (int r = 0; r < A.Rows(); ++r) {
    for (int c = 0; c < A.Cols(); ++c) {
      if (A(r, c) != B(r, c)) {
        return false;
      }
    }
  }
  return true;
}

TEST(Matrix, MatrixMove) {
  using namespace mochi::krylov;
  TestAllocator::ResetCounters();
  TestAllocator alloc1;
  TestAllocator alloc2;
  TestAllocator alloc3;

  // Fixed size matrix. Values get copied.
  {
    Matrix<real, 3, 3> a, b, c;
    a.SetRandom(123);
    b = a; // Copy the values
    c = std::move(a); // Can't move. Copy the values.
    EXPECT_TRUE(AreSame(a, b)); // NOLINT(bugprone-use-after-move)
    EXPECT_TRUE(AreSame(a, c)); // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(0, TestAllocator::s_allocate); // no allocations
  }

  // Dynamic matrix with default allocator. Pointer gets moved.
  {
    Matrix<real, kDynamic, 3> a(17, 3), b(17, 3), c(57, 3);
    a.SetRandom(123);
    b = a; // Copy the values
    auto* movedPtr = a.data();
    c = std::move(a); // Change dimensions of c and move pointer
    EXPECT_EQ(movedPtr, c.data()); // NOLINT(bugprone-use-after-move)
    EXPECT_EQ((decltype(c.data()))nullptr, a.data()); // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(17, b.Rows());
    EXPECT_EQ(17, c.Rows());
    EXPECT_TRUE(AreSame(b, c));
    EXPECT_TRUE(GetDefaultAllocator()->is_equal(*a.GetAllocator()));
    EXPECT_TRUE(GetDefaultAllocator()->is_equal(*b.GetAllocator()));
    EXPECT_TRUE(GetDefaultAllocator()->is_equal(*c.GetAllocator()));
  }

  // Dynamic matrix with compatible allocators. Pointer gets moved.
  {
    TestAllocator::s_compatibleWithOtherInstances = true;
    EXPECT_TRUE(alloc1.is_equal(alloc2));
    EXPECT_TRUE(alloc1.is_equal(alloc3));

    Matrix<real, kDynamic, 3> a(17, 3, &alloc1), b(17, 3, &alloc2), c(57, 3, &alloc3);
    EXPECT_EQ(3, TestAllocator::s_allocate);
    EXPECT_EQ(0, TestAllocator::s_deallocate);
    a.SetRandom(123);
    b = a; // Copy the values
    auto* movedPtr = a.data();
    c = std::move(a); // Move ownershipo of the pointer
    EXPECT_EQ(movedPtr, c.data()); // NOLINT(bugprone-use-after-move)
    EXPECT_EQ((decltype(a.data()))nullptr, a.data()); // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(17, c.Rows()); // new size
    EXPECT_EQ(movedPtr, c.data()); // no change
    EXPECT_TRUE(AreSame(b, c));
    EXPECT_EQ(&alloc1, a.GetAllocator()); // no change
    EXPECT_EQ(&alloc2, b.GetAllocator()); // no change
    EXPECT_EQ(&alloc3, c.GetAllocator()); // no change
    EXPECT_EQ(3, TestAllocator::s_allocate); // no change
    EXPECT_EQ(1, TestAllocator::s_deallocate); // Memory owned by c was freed
  }

  // Expect proper cleanup
  EXPECT_EQ(3, TestAllocator::s_allocate);
  EXPECT_EQ(3, TestAllocator::s_deallocate);
  EXPECT_EQ(0, TestAllocator::s_bytes);
  TestAllocator::ResetCounters();

  // Dynamic matrix with incompatible allocators. Values get copied.
  {
    TestAllocator::s_compatibleWithOtherInstances = false;
    EXPECT_FALSE(alloc1.is_equal(alloc2));
    EXPECT_FALSE(alloc1.is_equal(alloc3));

    Matrix<real, kDynamic, 3> a(17, 3, &alloc1), b(17, 3, &alloc2), c(57, 3, &alloc3);
    EXPECT_EQ(3, TestAllocator::s_allocate);
    EXPECT_EQ(0, TestAllocator::s_deallocate);
    a.SetRandom(123);
    b = a; // Copy the values
    auto* adata = a.data();
    c = std::move(a); // Memory can't be moved. Resize c using its own allocator, then copy values.
    EXPECT_EQ(adata, a.data()); // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(17, c.Rows()); // new size
    EXPECT_TRUE(AreSame(b, c));
    EXPECT_EQ(&alloc1, a.GetAllocator()); // no change
    EXPECT_EQ(&alloc2, b.GetAllocator()); // no change
    EXPECT_EQ(&alloc3, c.GetAllocator()); // no change
    EXPECT_EQ(4, TestAllocator::s_allocate); // New memory was allocated for c
    EXPECT_EQ(1, TestAllocator::s_deallocate); // Memory previously owned by c was freed
    c = std::move(b); // Memory can't be moved. Same size. Just copy values.
    EXPECT_EQ(adata, a.data()); // no change
    EXPECT_EQ(17, c.Rows()); // no change
    EXPECT_TRUE(AreSame(b, c));
    EXPECT_EQ(&alloc1, a.GetAllocator()); // no change
    EXPECT_EQ(&alloc2, b.GetAllocator()); // no change
    EXPECT_EQ(&alloc3, c.GetAllocator()); // no change
    EXPECT_EQ(4, TestAllocator::s_allocate); // no change
    EXPECT_EQ(1, TestAllocator::s_deallocate); // no change
  }

  // Expect proper cleanup
  EXPECT_EQ(4, TestAllocator::s_allocate);
  EXPECT_EQ(4, TestAllocator::s_deallocate);
  EXPECT_EQ(0, TestAllocator::s_bytes);
  TestAllocator::ResetCounters();
  TestAllocator::s_compatibleWithOtherInstances = true;

  // Can't move pointer to or from a view matrix. Values get copied.
  {
    Matrix<real> a(17, 3, &alloc1), b(17, 3, &alloc1);
    EXPECT_EQ(2, TestAllocator::s_allocate);
    EXPECT_EQ(0, TestAllocator::s_deallocate);
    a.SetRandom(123);
    b.SetRandom(456);
    EXPECT_FALSE(AreSame(b, a));
    // std::move makes no difference. Copy the values.
    b = std::move(AsView(a)); // NOLINT(performance-move-const-arg)
    EXPECT_TRUE(AreSame(b, a)); // NOLINT(bugprone-use-after-move)
    b.SetRandom(789);
    EXPECT_FALSE(AreSame(b, a));
    // Again, std::move makes no difference. Copy the values
    AsView(b) = std::move(a); // NOLINT(performance-move-const-arg)
    EXPECT_TRUE(AreSame(b, a)); // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(2, TestAllocator::s_allocate); // no change
    EXPECT_EQ(0, TestAllocator::s_deallocate); // no change
  }
}

TEST(Matrix, Swap) {
  // TODO(T143684599): Move assignment of views copies the data. This leads to undesired behavior
  // when std::swap is called on views. Example:
  //
  // T temp = std::move(Av); // temp is a view to A.
  // Av = std::move(Bv);     // Copies data from B to A. A and B now contain the same data.
  // Bv = std::move(temp);   // Copies data from A to B. A and B continue to contain the same data.

  constexpr int N = 10;

  // Dynamic size. Data is not in the object. Swap must not copy the data.
  {
    using MatType = Matrix<real>;
    MatType A0(N, 2 * N), B0(3 * N, 4 * N); // Different sizes to increase test coverage.
    A0.SetRandom(1);
    B0.SetRandom(2);

    MatType A = A0;
    MatType B = B0;
    auto const* aPt = A.Data();
    auto const* bPt = B.Data();
    auto Av = AsView(A);
    auto Bv = AsConstView(B);
    std::swap(A, B);
    EXPECT_TRUE(test::NearEqualMatrices(A0, B));
    EXPECT_TRUE(test::NearEqualMatrices(B0, A));
    EXPECT_EQ(A.Data(), bPt); // Did not copy the data.
    EXPECT_EQ(B.Data(), aPt); // Did not copy the data.
    EXPECT_TRUE(test::NearEqualMatrices(A0, Av)); // View still points to the correct data.
    EXPECT_TRUE(test::NearEqualMatrices(B0, Bv)); // View still points to the correct data.
  }

  // Compile-time size. Data is in the object. Swap copies the data.
  {
    using MatType = Matrix<real, N, 2 * N>;
    MatType A0, B0;
    A0.SetRandom(1);
    B0.SetRandom(2);

    MatType A = A0;
    MatType B = B0;
    std::swap(A, B);
    EXPECT_TRUE(test::NearEqualMatrices(A0, B));
    EXPECT_TRUE(test::NearEqualMatrices(B0, A));
  }
}

TEST(Matrix, FixedSizeMatrixMemory) {
  static_assert(
      sizeof(Matrix<real, 3, 5>) == sizeof(real) * 3 * 5, "Matrix has extra memory requirement");
}

TEST(Matrix, Constructors) {
  TestAllocator alloc;

  {
    Matrix<real, 3, 2> K;
    EXPECT_EQ(K.Rows(), 3);
    EXPECT_EQ(K.Cols(), 2);
    EXPECT_EQ(K.LeadDim(), 3);
    EXPECT_EQ((Allocator*)nullptr, K.GetAllocator()); // not dynamic
  }
  {
    RowVector<real> K(2);
    EXPECT_EQ(K.Rows(), 1);
    EXPECT_EQ(K.Cols(), 2);
    EXPECT_EQ(K.LeadDim(), 2);
    EXPECT_TRUE(GetDefaultAllocator()->is_equal(*K.GetAllocator()));
  }
  {
    TestAllocator::ResetCounters();
    RowVector<real> K(2, &alloc);
    EXPECT_EQ(K.Rows(), 1);
    EXPECT_EQ(K.Cols(), 2);
    EXPECT_EQ(K.LeadDim(), 2);
    EXPECT_EQ(&alloc, K.GetAllocator());
    EXPECT_EQ(1, TestAllocator::s_allocate);
    EXPECT_EQ(0, TestAllocator::s_deallocate);
    EXPECT_EQ(2 * sizeof(real), TestAllocator::s_bytes);
    EXPECT_EQ(alignof(real), TestAllocator::s_lastAllocAlignment);
  }
  EXPECT_EQ(1, TestAllocator::s_allocate);
  EXPECT_EQ(1, TestAllocator::s_deallocate);
  EXPECT_EQ(0, TestAllocator::s_bytes);
  EXPECT_EQ(alignof(real), TestAllocator::s_lastDeallocAlignment);
  {
    ColumnVector<real> K(3);
    EXPECT_EQ(K.Rows(), 3);
    EXPECT_EQ(K.Cols(), 1);
    EXPECT_EQ(K.LeadDim(), 3);
    EXPECT_TRUE(GetDefaultAllocator()->is_equal(*K.GetAllocator()));
  }
  {
    TestAllocator::ResetCounters();
    ColumnVector<real> K(3, &alloc);
    EXPECT_EQ(K.Rows(), 3);
    EXPECT_EQ(K.Cols(), 1);
    EXPECT_EQ(K.LeadDim(), 3);
    EXPECT_EQ(&alloc, K.GetAllocator());
    EXPECT_EQ(1, TestAllocator::s_allocate);
    EXPECT_EQ(0, TestAllocator::s_deallocate);
    EXPECT_EQ(3 * sizeof(real), TestAllocator::s_bytes);
    EXPECT_EQ(alignof(real), TestAllocator::s_lastAllocAlignment);
  }
  EXPECT_EQ(1, TestAllocator::s_allocate);
  EXPECT_EQ(1, TestAllocator::s_deallocate);
  EXPECT_EQ(0, TestAllocator::s_bytes);
  EXPECT_EQ(alignof(real), TestAllocator::s_lastDeallocAlignment);
  {
    Matrix<real, 4, 3> K(4, 3);
    EXPECT_EQ(K.Rows(), 4);
    EXPECT_EQ(K.Cols(), 3);
    EXPECT_EQ(K.LeadDim(), 4);
    EXPECT_EQ((Allocator*)nullptr, K.GetAllocator()); // not dynamic
  }
  {
    RowMatrix<real, 4, 3> K(4, 3);
    EXPECT_EQ(K.Rows(), 4);
    EXPECT_EQ(K.Cols(), 3);
    EXPECT_EQ(K.LeadDim(), 3);
    EXPECT_EQ((Allocator*)nullptr, K.GetAllocator()); // not dynamic
  }
  {
    Matrix<real, krylov::kDynamic, krylov::kDynamic> K(4, 3);
    EXPECT_EQ(K.Rows(), 4);
    EXPECT_EQ(K.Cols(), 3);
    EXPECT_EQ(K.LeadDim(), 4);
    EXPECT_TRUE(GetDefaultAllocator()->is_equal(*K.GetAllocator()));
  }
  {
    TestAllocator::ResetCounters();
    Matrix<real, krylov::kDynamic, krylov::kDynamic> K(4, 3, &alloc);
    EXPECT_EQ(K.Rows(), 4);
    EXPECT_EQ(K.Cols(), 3);
    EXPECT_EQ(K.LeadDim(), 4);
    EXPECT_EQ(&alloc, K.GetAllocator());
    EXPECT_EQ(1, TestAllocator::s_allocate);
    EXPECT_EQ(0, TestAllocator::s_deallocate);
    EXPECT_EQ(4 * 3 * sizeof(real), TestAllocator::s_bytes);
    EXPECT_EQ(alignof(real), TestAllocator::s_lastAllocAlignment);
  }
  {
    RowMatrix<real> K(4, 3);
    EXPECT_EQ(K.Rows(), 4);
    EXPECT_EQ(K.Cols(), 3);
    EXPECT_EQ(K.LeadDim(), 3);
    EXPECT_TRUE(GetDefaultAllocator()->is_equal(*K.GetAllocator()));
  }
  {
    TestAllocator::ResetCounters();
    RowMatrix<real> K(4, 3, &alloc);
    EXPECT_EQ(K.Rows(), 4);
    EXPECT_EQ(K.Cols(), 3);
    EXPECT_EQ(K.LeadDim(), 3);
    EXPECT_EQ(&alloc, K.GetAllocator());
    EXPECT_EQ(1, TestAllocator::s_allocate);
    EXPECT_EQ(0, TestAllocator::s_deallocate);
    EXPECT_EQ(4 * 3 * sizeof(real), TestAllocator::s_bytes);
    EXPECT_EQ(alignof(real), TestAllocator::s_lastAllocAlignment);
  }
  // Constructor with leading dimension and allocator.
  {
    TestAllocator::ResetCounters();
    Matrix<
        real,
        krylov::kDynamic,
        krylov::kDynamic,
        krylov::Direction::ColMajor,
        krylov::Ownership::Owner,
        krylov::kDynamic>
        K(4, 3, 6, &alloc);
    EXPECT_EQ(K.Rows(), 4);
    EXPECT_EQ(K.Cols(), 3);
    EXPECT_EQ(K.LeadDim(), 6);
    EXPECT_EQ(&alloc, K.GetAllocator());
    EXPECT_EQ(1, TestAllocator::s_allocate);
  }
  // Span constructor (views only).
  {
    real data[] = {1_r, 2_r, 3_r, 4_r, 5_r};
    Span<real> sp(data);
    ColumnVectorView<real> vec(sp);
    EXPECT_EQ(vec.Rows(), 5);
    EXPECT_EQ(vec.Data(), data);
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(vec[i], data[i]);
    }
  }
  {
    Matrix<real, 4, 3> K;
    Matrix<real> M(2.3f * K);
    EXPECT_EQ(M.Rows(), 4);
    EXPECT_EQ(M.Cols(), 3);
    EXPECT_EQ(M.LeadDim(), 4);
    EXPECT_EQ((Allocator*)nullptr, K.GetAllocator()); // not dynamic
    EXPECT_TRUE(GetDefaultAllocator()->is_equal(*M.GetAllocator()));
  }
  {
    TestAllocator::ResetCounters();
    Matrix<real, 4, 3> K;
    Matrix<real> M(2.3f * K, &alloc);
    EXPECT_EQ(M.Rows(), 4);
    EXPECT_EQ(M.Cols(), 3);
    EXPECT_EQ(M.LeadDim(), 4);
    EXPECT_EQ((Allocator*)nullptr, K.GetAllocator()); // not dynamic
    EXPECT_EQ(&alloc, M.GetAllocator());
    EXPECT_EQ(1, TestAllocator::s_allocate);
    EXPECT_EQ(0, TestAllocator::s_deallocate);
    EXPECT_EQ(4 * 3 * sizeof(real), TestAllocator::s_bytes);
    EXPECT_EQ(alignof(real), TestAllocator::s_lastAllocAlignment);
  }
  {
    Matrix<real, 2, 3> K1;
    Matrix<real, 2, 3> K2;
    Matrix<real> M(K1 + K2);
    EXPECT_EQ(M.Rows(), 2);
    EXPECT_EQ(M.Cols(), 3);
    EXPECT_EQ(M.LeadDim(), 2);
    EXPECT_EQ((Allocator*)nullptr, K1.GetAllocator()); // not dynamic
    EXPECT_EQ((Allocator*)nullptr, K2.GetAllocator()); // not dynamic
    EXPECT_TRUE(GetDefaultAllocator()->is_equal(*M.GetAllocator()));
  }
  {
    TestAllocator::ResetCounters();
    Matrix<real, 2, 3> K1;
    Matrix<real, 2, 3> K2;
    Matrix<real> M(K1 + K2, &alloc);
    EXPECT_EQ(M.Rows(), 2);
    EXPECT_EQ(M.Cols(), 3);
    EXPECT_EQ(M.LeadDim(), 2);
    EXPECT_EQ((Allocator*)nullptr, K1.GetAllocator()); // not dynamic
    EXPECT_EQ((Allocator*)nullptr, K2.GetAllocator()); // not dynamic
    EXPECT_EQ(&alloc, M.GetAllocator());
    EXPECT_EQ(1, TestAllocator::s_allocate);
    EXPECT_EQ(0, TestAllocator::s_deallocate);
    EXPECT_EQ(2 * 3 * sizeof(real), TestAllocator::s_bytes);
    EXPECT_EQ(alignof(real), TestAllocator::s_lastAllocAlignment);
  }
  {
    Matrix<real, 2, 3> K1;
    Matrix<real, 3, 4> K2;
    Matrix<real> M(K1 * K2);
    EXPECT_EQ(M.Rows(), 2);
    EXPECT_EQ(M.Cols(), 4);
    EXPECT_EQ(M.LeadDim(), 2);
    EXPECT_EQ((Allocator*)nullptr, K1.GetAllocator()); // not dynamic
    EXPECT_EQ((Allocator*)nullptr, K2.GetAllocator()); // not dynamic
    EXPECT_TRUE(GetDefaultAllocator()->is_equal(*M.GetAllocator()));
  }
  {
    Matrix<real, 2, 3> K1;
    Matrix<real, 3, 4> K2;
    Matrix<real, 2, 4> K3;
    Matrix<real> M = -2.3_r * K1 * K2 + 1.2_r * K3;
    EXPECT_EQ(M.Rows(), 2);
    EXPECT_EQ(M.Cols(), 4);
    EXPECT_EQ(M.LeadDim(), 2);
    EXPECT_EQ((Allocator*)nullptr, K1.GetAllocator()); // not dynamic
    EXPECT_EQ((Allocator*)nullptr, K2.GetAllocator()); // not dynamic
    EXPECT_EQ((Allocator*)nullptr, K3.GetAllocator()); // not dynamic
    EXPECT_TRUE(GetDefaultAllocator()->is_equal(*M.GetAllocator()));
  }
  {
    TestAllocator::ResetCounters();
    Matrix<real, 2, 3> K1;
    Matrix<real, 3, 4> K2;
    Matrix<real, 2, 4> K3;
    Matrix<real> M(-2.3_r * K1 * K2 + 1.2_r * K3, &alloc);
    EXPECT_EQ(M.Rows(), 2);
    EXPECT_EQ(M.Cols(), 4);
    EXPECT_EQ(M.LeadDim(), 2);
    EXPECT_EQ((Allocator*)nullptr, K1.GetAllocator()); // not dynamic
    EXPECT_EQ((Allocator*)nullptr, K2.GetAllocator()); // not dynamic
    EXPECT_EQ((Allocator*)nullptr, K3.GetAllocator()); // not dynamic
    EXPECT_EQ(&alloc, M.GetAllocator());
    EXPECT_EQ(1, TestAllocator::s_allocate);
    EXPECT_EQ(0, TestAllocator::s_deallocate);
    EXPECT_EQ(2 * 4 * sizeof(real), TestAllocator::s_bytes);
    EXPECT_EQ(alignof(real), TestAllocator::s_lastAllocAlignment);
  }
  {
    Matrix<real, 2, 3> K1;
    MatrixView<real, 2, 3> M(K1);
    EXPECT_EQ(M.Rows(), 2);
    EXPECT_EQ(M.Cols(), 3);
    EXPECT_EQ(M.LeadDim(), 2);
    EXPECT_EQ((Allocator*)nullptr, K1.GetAllocator()); // not dynamic
    EXPECT_EQ((Allocator*)nullptr, M.GetAllocator()); // not owning
  }
  {
    Matrix<real, krylov::kDynamic, krylov::kDynamic> K1(2, 3);
    MatrixView<real> M(K1);
    EXPECT_EQ(M.Rows(), 2);
    EXPECT_EQ(M.Cols(), 3);
    EXPECT_EQ(M.LeadDim(), 2);
    EXPECT_TRUE(GetDefaultAllocator()->is_equal(*K1.GetAllocator()));
    EXPECT_EQ((Allocator*)nullptr, M.GetAllocator()); // not owning
  }
  {
    Matrix<real, 2, 3> K1;
    MatrixView<real> M(K1);
    EXPECT_EQ(M.Rows(), 2);
    EXPECT_EQ(M.Cols(), 3);
    EXPECT_EQ(M.LeadDim(), 2);
    EXPECT_EQ((Allocator*)nullptr, K1.GetAllocator()); // not dynamic
    EXPECT_EQ((Allocator*)nullptr, M.GetAllocator()); // not owning
  }
  {
    constexpr int nr = 3, nc = 5;
    real a[nr * nc];
    auto aview = MatrixView<real, nr, nc>(&a[0]);
    aview.SetConstant(1_r);
    EXPECT_EQ(aview(1, 2), 1_r);
    Matrix<real, nr, nc> dstMatrix(aview);
    EXPECT_EQ(dstMatrix(1, 2), 1_r);
    EXPECT_EQ((Allocator*)nullptr, aview.GetAllocator()); // not owning
  }
  {
    // Copy constructor, fixed-size
    Matrix<real, 3, 3> a;
    a.SetRandom(123);
    Matrix<real, 3, 3> b(a);
    AreSame(a, b);
  }
  {
    // Copy constructor, dynamic
    Matrix<real> a(3, 3);
    a.SetRandom(123);
    Matrix<real> b(a);
    AreSame(a, b);
    EXPECT_TRUE(GetDefaultAllocator()->is_equal(*a.GetAllocator()));
    EXPECT_TRUE(GetDefaultAllocator()->is_equal(*b.GetAllocator()));
  }
  {
    // Copy constructor, dynamic + allocator
    TestAllocator::ResetCounters();
    Matrix<real> a(3, 3);
    a.SetRandom(123);
    Matrix<real> b(a, &alloc);
    AreSame(a, b);
    EXPECT_EQ(1, TestAllocator::s_allocate);
    EXPECT_EQ(0, TestAllocator::s_deallocate);
    EXPECT_EQ(sizeof(real) * 3 * 3, TestAllocator::s_bytes);
    EXPECT_TRUE(GetDefaultAllocator()->is_equal(*a.GetAllocator()));
    EXPECT_EQ(&alloc, b.GetAllocator());
  }
  EXPECT_EQ(1, TestAllocator::s_allocate);
  EXPECT_EQ(1, TestAllocator::s_deallocate);
  EXPECT_EQ(0, TestAllocator::s_bytes);
  {
    // Move constructor, fixed-size
    Matrix<real, 3, 3> a;
    a.SetRandom(123);
    auto aCopy = a;
    Matrix<real, 3, 3> b(std::move(a));
    AreSame(aCopy, b);
  }
  {
    // Move constructor, dynamic
    Matrix<real> a(3, 3);
    EXPECT_TRUE(GetDefaultAllocator()->is_equal(*a.GetAllocator()));
    a.SetRandom(123);
    auto aCopy = a;
    auto* movedPtr = a.data();
    Matrix<real> b(std::move(a));
    EXPECT_TRUE(GetDefaultAllocator()->is_equal(*b.GetAllocator()));
    AreSame(aCopy, b);
    EXPECT_EQ(movedPtr, b.data());
    EXPECT_EQ((real*)nullptr, a.data()); // NOLINT(bugprone-use-after-move)
  }
  {
    // Move constructor, dynamic + incompatible allocators
    TestAllocator::ResetCounters();
    TestAllocator::s_compatibleWithOtherInstances = false;
    TestAllocator alloc2;
    Matrix<real> a(3, 3, &alloc);
    EXPECT_EQ(&alloc, a.GetAllocator());
    a.SetRandom(123);
    auto aCopy = a;
    Matrix<real> b(std::move(a), &alloc2);
    EXPECT_EQ(&alloc2, b.GetAllocator());
    AreSame(aCopy, b);
    AreSame(a, b); // NOLINT(bugprone-use-after-move)
  }
  {
    // Move constructor, dynamic + compatible allocators
    TestAllocator::ResetCounters();
    TestAllocator::s_compatibleWithOtherInstances = true;
    TestAllocator alloc2;
    Matrix<real> a(3, 3, &alloc);
    EXPECT_EQ(&alloc, a.GetAllocator());
    a.SetRandom(123);
    auto aCopy = a;
    auto* movedPtr = a.data();
    Matrix<real> b(std::move(a), &alloc2);
    EXPECT_EQ(&alloc2, b.GetAllocator());
    AreSame(aCopy, b);
    EXPECT_EQ(movedPtr, b.data());
    EXPECT_EQ((real*)nullptr, a.data()); // NOLINT(bugprone-use-after-move)
  }
}

TEST(Matrix, VectorNormalized) {
  ColumnVector<real, 3> v{3_r, 0_r, 4_r};
  auto vn = Normalized(v);
  EXPECT_NEAR_EQ(vn.Norm(), 1_r);
  EXPECT_NEAR_EQ(vn[0], 0.6_r);
  EXPECT_NEAR_EQ(vn[2], 0.8_r);

  // Rvalue Normalize() — exercises the && overload.
  auto n = ColumnVector<real, 3>{3_r, 0_r, 4_r}.Normalize();
  EXPECT_NEAR_EQ(n.Norm(), 1_r);
  EXPECT_NEAR_EQ(n[0], 0.6_r);
  EXPECT_NEAR_EQ(n[2], 0.8_r);
}

TEST(Matrix, CrossProduct) {
  Matrix<real, 3, 1> v{2_r, 3_r, 4_r};
  Matrix<real, 3, 1> w{4_r, 1_r, 9_r};
  auto c = Cross(v, w);
  auto sc = Cross(c, c);
  auto vsqn = v.Dot(v);
  auto wsqn = w.Dot(w);
  auto dvw = v.Dot(w);
  auto csqn = c.Dot(c);
  EXPECT_NEAR_EQ(c.Dot(v), 0_r); // c is orthogonal to v
  EXPECT_NEAR_EQ(c.Dot(w), 0_r); // c is orthogonal to w
  EXPECT_NEAR_EQ(sc.Norm(), 0_r); // Crossproduct to itself is zero
  EXPECT_NEAR_EQ(csqn, vsqn * wsqn - dvw * dvw); // || c || = ||a|| ||b|| sin(a,b)
}

TEST(Matrix, Resize) {
  {
    Matrix<real> M(2, 5);
    M.Resize(12, 64);
    EXPECT_EQ(M.Rows(), 12);
    EXPECT_EQ(M.Cols(), 64);
    auto const* ptr1 = &M(0, 0);
    M.Resize(12, 64); // Same size
    EXPECT_EQ(M.Rows(), 12);
    EXPECT_EQ(M.Cols(), 64);
    auto const* ptr2 = &M(0, 0);
    EXPECT_EQ(ptr1, ptr2); // Expect same address
  }
  {
    ColumnVector<real> v;
    v.Resize(32);
    EXPECT_EQ(v.Rows(), 32);
    EXPECT_EQ(v.Cols(), 1);
    auto const* ptr1 = v.data();
    v.Resize(32); // Same size
    EXPECT_EQ(v.Rows(), 32);
    EXPECT_EQ(v.Cols(), 1);
    auto const* ptr2 = v.data();
    EXPECT_EQ(ptr1, ptr2); // Expect same address
  }

  {
    RowVector<real> w(6);
    w.Resize(56);
    EXPECT_EQ(w.Rows(), 1);
    EXPECT_EQ(w.Cols(), 56);
    auto const* ptr1 = w.data();
    w.Resize(56); // Same size
    EXPECT_EQ(w.Rows(), 1);
    EXPECT_EQ(w.Cols(), 56);
    auto const* ptr2 = w.data();
    EXPECT_EQ(ptr1, ptr2); // Expect same address
  }

  {
    Matrix<real> A(5, 12), B(12, 19), M(1, 1);
    A.SetRandom(5);
    B.SetRandom(12);
    M = A * B; // automatic Resize on assignment
    EXPECT_EQ(M.Rows(), 5);
    EXPECT_EQ(M.Cols(), 19);
    auto const* ptr1 = &M(0, 0);
    A.SetRandom(123);
    B.SetRandom(456);
    M = A * B;
    EXPECT_EQ(M.Rows(), 5);
    EXPECT_EQ(M.Cols(), 19);
    auto const* ptr2 = &M(0, 0);
    EXPECT_EQ(ptr1, ptr2); // Expect same address
  }
}

TEST(Matrix, Reset) {
  ColumnVector<real> v(10);
  ColumnVectorView<real> w;
  ColumnVectorView<real const> z;
  for (int i = 0; i < v.Rows(); ++i) {
    v[i] = real(i - 1);
  }
  w.Reset(v.data() + 2, 5);
  EXPECT_EQ(w.Rows(), 5);
  EXPECT_EQ(w[0], 1);
  EXPECT_EQ(w[4], 5);
  z.Reset(v.data() + 3, 6);
  EXPECT_EQ(z.Rows(), 6);
  EXPECT_EQ(z[0], 2);
  EXPECT_EQ(z[5], 7);
  auto y = w.GetConstSpan();
  auto t = z.GetSpan();
  static_assert(std::is_same_v<decltype(y[0]), decltype(t[0])>, "Unexpected types");
  EXPECT_EQ(y[0], 1);
  EXPECT_EQ(t[0], 2);
}

TEST(Matrix, MatrixHoldsData) {
  // Test whether the matrix holds (or can store) data
  Matrix<real> A(256, 35);
  EXPECT_TRUE(A);
  Matrix<real, 3, 2> B;
  EXPECT_TRUE(B);
  Matrix<real> C;
  EXPECT_FALSE(C);
  Matrix<real, 3> D;
  EXPECT_FALSE(D);
  Matrix<real, krylov::kDynamic, 3> E;
  EXPECT_FALSE(E);
  //--- Matrix view
  MatrixView<real> F;
  EXPECT_FALSE(F);
  std::vector<real> values(6);
  MatrixView<real> G(values.data(), 2, 3);
  EXPECT_TRUE(G);
}

TEST(Matrix, SetIdentity) {
  auto IsIdentity = [](auto& M) -> bool {
    bool isIdentity = true;
    isIdentity &= (M.Rows() == M.Cols());
    for (int r = 0; r < M.Rows(); ++r) {
      for (int c = 0; c < M.Cols(); ++c) {
        isIdentity &= (M(r, c) == (r == c ? 1 : 0));
      }
    }
    return isIdentity;
  };

  // Dynamic, col-major
  Matrix<real> M1(10, 10);
  M1.SetRandom(0);
  EXPECT_FALSE(IsIdentity(M1));
  M1.SetIdentity(); // Set from owner
  EXPECT_TRUE(IsIdentity(M1));
  M1.SetRandom(1);
  EXPECT_FALSE(IsIdentity(M1));
  auto M1v = AsView(M1);
  M1v.SetIdentity(); // Set from view
  EXPECT_TRUE(IsIdentity(M1)); // Check from onwner

  // Dynamic, row-major
  RowMatrix<real> M2(12, 12);
  M2.SetRandom(2);
  EXPECT_FALSE(IsIdentity(M2));
  M2.SetIdentity(); // Set from owner
  EXPECT_TRUE(IsIdentity(M2));
  M2.SetRandom(3);
  EXPECT_FALSE(IsIdentity(M2));
  auto M2v = AsView(M2);
  M2v.SetIdentity(); // Set from view
  EXPECT_TRUE(IsIdentity(M2v)); // Check from view

  // Compile-time, col-major
  Matrix<real, 6, 6> M3(6, 6);
  M3.SetRandom(4);
  EXPECT_FALSE(IsIdentity(M3));
  M3.SetIdentity(); // Set from owner
  EXPECT_TRUE(IsIdentity(M3));
  M3.SetRandom(5);
  EXPECT_FALSE(IsIdentity(M3));
  auto M3v = AsView(M3);
  M3v.SetIdentity(); // Set from view
  EXPECT_TRUE(IsIdentity(M3v)); // Check from view

  // Compile-time, row-major
  RowMatrix<real, 21, 21> M4(21, 21);
  M4.SetRandom(6);
  EXPECT_FALSE(IsIdentity(M4));
  M4.SetIdentity(); // Set from owner
  EXPECT_TRUE(IsIdentity(M4));
  M4.SetRandom(7);
  EXPECT_FALSE(IsIdentity(M4));
  auto M4v = AsView(M4);
  M4v.SetIdentity(); // Set from view
  EXPECT_TRUE(IsIdentity(M4)); // Check from onwner
}

TEST(Matrix, Zero) {
  // Dynamic matrix.
  EXPECT_EQ(0_r, ColumnVector<real>::Zero(10).Norm());
  EXPECT_EQ(0_r, RowVector<real>::Zero(10).Norm());
  EXPECT_EQ(0_r, Matrix<real>::Zero(10, 12).Norm());
  EXPECT_EQ(0_r, RowMatrix<real>::Zero(10, 12).Norm());

  // Fixed-size matrix.
  EXPECT_EQ(0_r, (ColumnVector<real, 10>::Zero()).Norm());
  EXPECT_EQ(0_r, (ColumnVector<real, 10>::Zero(10)).Norm());
  EXPECT_EQ(0_r, (RowVector<real, 10>::Zero()).Norm());
  EXPECT_EQ(0_r, (RowVector<real, 10>::Zero(10)).Norm());
  EXPECT_EQ(0_r, (Matrix<real, 10, 12>::Zero()).Norm());
  EXPECT_EQ(0_r, (Matrix<real, 10, 12>::Zero(10, 12)).Norm());
  EXPECT_EQ(0_r, (RowMatrix<real, 10, 12>::Zero()).Norm());
  EXPECT_EQ(0_r, (RowMatrix<real, 10, 12>::Zero(10, 12)).Norm());
}

TEST(Matrix, Ones) {
  auto checkOnes = [](auto const& A, int nRows, int nCols) {
    EXPECT_EQ(nRows, A.Rows());
    EXPECT_EQ(nCols, A.Cols());
    for (int i = 0; i < A.Rows(); ++i) {
      for (int j = 0; j < A.Cols(); ++j) {
        EXPECT_EQ(A(i, j), 1_r);
      }
    }
  };

  // Dynamic matrix.
  checkOnes(ColumnVector<real>::Ones(10), 10, 1);
  checkOnes(RowVector<real>::Ones(10), 1, 10);
  checkOnes(Matrix<real>::Ones(10, 12), 10, 12);
  checkOnes(RowMatrix<real>::Ones(10, 12), 10, 12);

  // Fixed-size matrix.
  checkOnes(ColumnVector<real, 10>::Ones(), 10, 1);
  checkOnes(ColumnVector<real, 10>::Ones(10), 10, 1);
  checkOnes(RowVector<real, 10>::Ones(), 1, 10);
  checkOnes(RowVector<real, 10>::Ones(10), 1, 10);
  checkOnes(Matrix<real, 10, 12>::Ones(), 10, 12);
  checkOnes(Matrix<real, 10, 12>::Ones(10, 12), 10, 12);
  checkOnes(RowMatrix<real, 10, 12>::Ones(), 10, 12);
  checkOnes(RowMatrix<real, 10, 12>::Ones(10, 12), 10, 12);
}

TEST(Matrix, Random) {
  real const sMin = -0.3_r;
  real const sMax = 2.3_r;
  auto checkRandom = [&](auto const& A, int nRows, int nCols) {
    EXPECT_EQ(nRows, A.Rows());
    EXPECT_EQ(nCols, A.Cols());
    EXPECT_GT(A.Norm(), 0_r); // Initializes to non-zero values.
    for (int i = 0; i < A.Rows(); ++i) {
      for (int j = 0; j < A.Cols(); ++j) {
        EXPECT_GE(A(i, j), sMin);
        EXPECT_LE(A(i, j), sMax);
      }
    }
  };

  checkRandom(ColumnVector<real, 10>::Random(1, sMin, sMax), 10, 1);
  checkRandom(RowVector<real, 10>::Random(2, sMin, sMax), 1, 10);
  checkRandom(Matrix<real, 10, 12>::Random(3, sMin, sMax), 10, 12);
  checkRandom(RowMatrix<real, 10, 12>::Random(4, sMin, sMax), 10, 12);
}

TEST(Matrix, SetConstant) {
  // Fixed-size matrix.
  Matrix<real, 3, 3> mat;
  mat.SetConstant(3.5_r);
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      EXPECT_EQ(mat(r, c), 3.5_r);
    }
  }

  // Dynamic matrix.
  Matrix<real> dyn(4, 5);
  dyn.SetConstant(-2_r);
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 5; ++c) {
      EXPECT_EQ(dyn(r, c), -2_r);
    }
  }

  // Row-major matrix.
  RowMatrix<real, 3, 4> rowMat;
  rowMat.SetConstant(7_r);
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 4; ++c) {
      EXPECT_EQ(rowMat(r, c), 7_r);
    }
  }
}

TEST(Matrix, InitializerList) {
  // Fixed sized owning matrices can be initialized directly from their values, specified in memory
  // storage order. This method uses a single pair of {} brackets. The values are passed to the
  // constructor as arguments.

  {
    ColumnVector<real, 3> vec = {1_r, 2_r, 3_r};
    EXPECT_EQ(vec[0], 1_r);
    EXPECT_EQ(vec[1], 2_r);
    EXPECT_EQ(vec[2], 3_r);
  }

  {
    RowVector<real, 4> vec = {1_r, 2_r, 3_r, 4_r};
    EXPECT_EQ(vec[0], 1_r);
    EXPECT_EQ(vec[1], 2_r);
    EXPECT_EQ(vec[2], 3_r);
    EXPECT_EQ(vec[3], 4_r);
  }

  {
    Matrix<real, 2, 3> mat = {1_r, 2_r, 3_r, 4_r, 5_r, 6_r};
    EXPECT_EQ(mat(0, 0), 1_r);
    EXPECT_EQ(mat(1, 0), 2_r);
    EXPECT_EQ(mat(0, 1), 3_r);
    EXPECT_EQ(mat(1, 1), 4_r);
    EXPECT_EQ(mat(0, 2), 5_r);
    EXPECT_EQ(mat(1, 2), 6_r);
  }

  {
    RowMatrix<real, 2, 3> mat = {1_r, 2_r, 3_r, 4_r, 5_r, 6_r};
    EXPECT_EQ(mat(0, 0), 1_r);
    EXPECT_EQ(mat(0, 1), 2_r);
    EXPECT_EQ(mat(0, 2), 3_r);
    EXPECT_EQ(mat(1, 0), 4_r);
    EXPECT_EQ(mat(1, 1), 5_r);
    EXPECT_EQ(mat(1, 2), 6_r);
  }

  // Matrices can also be initialized using nested {{}} brackets, which provide a list of rows (if
  // row-major) or a list of columns (if col-major). The values will be passed to the constructor
  // using nested std::initializer_list. This syntax also works for dynamic matrices.

  {
    ColumnVector<real, 3> vec = {{1_r, 2_r, 3_r}};
    EXPECT_EQ(vec[0], 1_r);
    EXPECT_EQ(vec[1], 2_r);
    EXPECT_EQ(vec[2], 3_r);
    ColumnVector<real> vec2 = {{1_r, 2_r, 3_r}}; // Dynamic
    EXPECT_TRUE(test::NearEqualMatrices(vec, vec2));
  }

  {
    RowVector<real, 4> vec = {{1_r, 2_r, 3_r, 4_r}};
    EXPECT_EQ(vec[0], 1_r);
    EXPECT_EQ(vec[1], 2_r);
    EXPECT_EQ(vec[2], 3_r);
    EXPECT_EQ(vec[3], 4_r);
    RowVector<real> vec2 = {{1_r, 2_r, 3_r, 4_r}}; // Dynamic
    EXPECT_TRUE(test::NearEqualMatrices(vec, vec2));
  }

  {
    Matrix<real, 2, 3> mat = {{1_r, 2_r}, {3_r, 4_r}, {5_r, 6_r}}; // From column
    EXPECT_EQ(mat(0, 0), 1_r);
    EXPECT_EQ(mat(1, 0), 2_r);
    EXPECT_EQ(mat(0, 1), 3_r);
    EXPECT_EQ(mat(1, 1), 4_r);
    EXPECT_EQ(mat(0, 2), 5_r);
    EXPECT_EQ(mat(1, 2), 6_r);
    Matrix<real> mat2 = {{1_r, 2_r}, {3_r, 4_r}, {5_r, 6_r}}; // Dynamic
    EXPECT_TRUE(test::NearEqualMatrices(mat, mat2));
  }

  {
    RowMatrix<real, 2, 3> mat = {{1_r, 2_r, 3_r}, {4_r, 5_r, 6_r}}; // From rows
    EXPECT_EQ(mat(0, 0), 1_r);
    EXPECT_EQ(mat(0, 1), 2_r);
    EXPECT_EQ(mat(0, 2), 3_r);
    EXPECT_EQ(mat(1, 0), 4_r);
    EXPECT_EQ(mat(1, 1), 5_r);
    EXPECT_EQ(mat(1, 2), 6_r);
    RowMatrix<real> mat2 = {{1_r, 2_r, 3_r}, {4_r, 5_r, 6_r}}; // Dynamic
    EXPECT_TRUE(test::NearEqualMatrices(mat, mat2));
  }
}
