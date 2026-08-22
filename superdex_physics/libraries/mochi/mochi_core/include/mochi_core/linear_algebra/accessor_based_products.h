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

#include <mochi_core/linear_algebra/base_tools.h>
#include <mochi_core/linear_algebra/matrix_accessors.h>
#include <mochi_core/memory/filo_allocator.h>
#include <mochi_core/mochi_config.h>
#include <mochi_core/mochi_platform.h>
#include <mochi_core/utils/simd.h>

#if MOCHI_USE_MATH_ACCELERATION && MOCHI_PLATFORM_MACOS
#include <Accelerate/Accelerate.h>
#endif

#include <algorithm>
#include <type_traits>

namespace mochi::details {

/// @brief Matrix-matrix multiplication kernel routine.
/// @details The kernel computes C = A * B, where C is N x 6 stored column major, A is N x nc stored
/// column major, B is nc x 6 with row stride of b_ri and column stride of b_ci, and N is twice the
/// size of the SIMD vector. The kernel is optimized for a SIMD vector size such that 16 vectors
/// fill the floating-point registers.
/// @tparam VType SIMD vector type to use.
/// @param a Pointer to the start of matrix A.
/// @param b Pointer to the start of matrix B.
/// @param c Pointer to the start of matrix C.
/// @param nc Number of columns of A and rows of B.
/// @param b_ri Row stride of B.
/// @param b_ci Column stride of B.
template <typename VType>
MOCHI_ANY void MatMatKernel(
    typename VType::Scalar const* a,
    typename VType::Scalar const* b,
    typename VType::Scalar* c,
    int nc,
    int b_ri,
    int b_ci);

template <typename Scalar>
struct Multiplier;

/// @brief Auxiliary kernel to compute rows [r0, r0 + kNumRows) of the output matrix C in
/// DirectProductSimdAlongM.
/// @remarks
/// - Requires a scalar type with SIMD support.
/// - Most performant for A column-major and (less critical) B and C also column-major.
template <
    int kNumRows,
    typename Scalar,
    typename AccessorC,
    typename AccessorA,
    typename AccessorB,
    int mATC,
    int nATC,
    int kATC,
    typename MultScalar>
MOCHI_ANY void DirectRowBlockProductSimdAlongM(
    AccessorC&& C,
    AccessorA const& A,
    AccessorB const& B,
    [[maybe_unused]] IntOrEmpty<mATC> m,
    IntOrEmpty<nATC> n,
    IntOrEmpty<kATC> k,
    Multiplier<MultScalar> factor,
    int r0) {
  // Use the smallest SIMD size that is greater than or equal to kNumRows.
  using VType = Simd<Scalar, kNextSupportedSimdSize<Scalar, kNumRows>>;
  for (int c = 0; c < n; ++c) {
    int l = 0;
    auto s = SimdZero<VType>();
    if constexpr (kATC == krylov::kDynamic || kATC >= 12) { // At least 3 batches.
      // Using batches of 4 SIMD vectors systematically improves performance on various x86-64 and
      // ARM architectures.
      for (; l + 4 <= k; l += 4) {
        VType s0 = A.template ColVector<VType, kNumRows>(r0, l + 0) * B(l + 0, c);
        VType s1 = A.template ColVector<VType, kNumRows>(r0, l + 1) * B(l + 1, c);
        VType s2 = A.template ColVector<VType, kNumRows>(r0, l + 2) * B(l + 2, c);
        VType s3 = A.template ColVector<VType, kNumRows>(r0, l + 3) * B(l + 3, c);
        s += s0 + s1 + s2 + s3;
      }
    }
    for (; l < k; ++l) {
      s += A.template ColVector<VType, kNumRows>(r0, l) * B(l, c);
    }
    if constexpr (std::is_same_v<MultScalar, void>) {
      C.template StoreColVector<kNumRows>(r0, c, s);
    } else {
      C.template StoreColVector<kNumRows>(r0, c, s * Scalar(factor.factor));
    }
  }
}

#define MOCHI_CASE_DIRECT_ROW_BLOCK_PRODUCT_SIMD_ALONG_M(NUM_ROWS)                  \
  case NUM_ROWS: {                                                                  \
    DirectRowBlockProductSimdAlongM<NUM_ROWS, Scalar>(C, A, B, m, n, k, factor, r); \
    break;                                                                          \
  }

/// @brief Matrix-matrix product using SIMD instructions along the "m" direction.
/// @remarks
/// - Requires a scalar type with SIMD support.
/// - Most performant for A column-major and (less critical) B and C also column-major.
/// @tparam Scalar
/// @tparam AccessorC
/// @tparam AccessorA
/// @tparam AccessorB
/// @tparam MultScalar
/// @tparam mATC
/// @tparam nATC
/// @tparam kATC
/// @param C
/// @param A
/// @param B
/// @param m
/// @param n
/// @param k
/// @param factor
template <
    typename Scalar,
    typename AccessorC,
    typename AccessorA,
    typename AccessorB,
    int mATC,
    int nATC,
    int kATC,
    typename MultScalar>
MOCHI_ANY void DirectProductSimdAlongM(
    AccessorC&& C,
    AccessorA const& A,
    AccessorB const& B,
    IntOrEmpty<mATC> m,
    IntOrEmpty<nATC> n,
    IntOrEmpty<kATC> k,
    Multiplier<MultScalar> factor) {
  static_assert(
      !std::is_const_v<Scalar>, "Implementation requires the scalar type to be non-const");
  static_assert(
      Simd<Scalar>::kIsSupported, "Implementation requires a scalar type with SIMD support");
  //--- Compute the product in batches of rows to increase the number of concurrent FMA instructions
  //--- and improve cache line utilization. The number of rows is a compile-time constant to enable
  //--- better compiler optimizations. The optimal number of rows depends on a number of factors,
  //--- such as scalar type, register size and cache line size. 32 rows per batch is empirically
  //--- optimal (or near-optimal) for single-precision arithmetic on several x86-64 and ARM
  //--- architectures.
  constexpr int kRowsPerBatch = 32;
  int r = 0;
  if constexpr (mATC == krylov::kDynamic || mATC >= kRowsPerBatch) {
    for (; r + kRowsPerBatch <= m; r += kRowsPerBatch) {
      DirectRowBlockProductSimdAlongM<kRowsPerBatch, Scalar>(C, A, B, m, n, k, factor, r);
    }
  }
  //--- Leftover rows.
  constexpr int kLeftoverRows = mATC % kRowsPerBatch;
  if constexpr (mATC > 0 && kLeftoverRows > 0) {
    static_assert(kLeftoverRows < kRowsPerBatch, "Inconsistent number of leftover rows");
    DirectRowBlockProductSimdAlongM<kLeftoverRows, Scalar>(C, A, B, m, n, k, factor, r);
  } else if constexpr (mATC == krylov::kDynamic) {
    //--- One batch of half the number of rows to reduce the number of cases in the switch below.
    //--- This slightly degrades performance but improves build time.
    constexpr int kRowsPerBatchHalf = kRowsPerBatch / 2;
    if (r + kRowsPerBatchHalf <= m) {
      DirectRowBlockProductSimdAlongM<kRowsPerBatchHalf, Scalar>(C, A, B, m, n, k, factor, r);
      r += kRowsPerBatchHalf;
    }
    int const leftOverRows = m.iVal() - r;
    MOCHI_ASSERT_VERBOSE(
        leftOverRows >= 0 && leftOverRows < kRowsPerBatchHalf,
        "Inconsistent number of leftover rows.");
    if (leftOverRows > 0) {
      switch (leftOverRows) {
        MOCHI_CASE_DIRECT_ROW_BLOCK_PRODUCT_SIMD_ALONG_M(1);
        MOCHI_CASE_DIRECT_ROW_BLOCK_PRODUCT_SIMD_ALONG_M(2);
        MOCHI_CASE_DIRECT_ROW_BLOCK_PRODUCT_SIMD_ALONG_M(3);
        MOCHI_CASE_DIRECT_ROW_BLOCK_PRODUCT_SIMD_ALONG_M(4);
        MOCHI_CASE_DIRECT_ROW_BLOCK_PRODUCT_SIMD_ALONG_M(5);
        MOCHI_CASE_DIRECT_ROW_BLOCK_PRODUCT_SIMD_ALONG_M(6);
        MOCHI_CASE_DIRECT_ROW_BLOCK_PRODUCT_SIMD_ALONG_M(7);
        MOCHI_CASE_DIRECT_ROW_BLOCK_PRODUCT_SIMD_ALONG_M(8);
        MOCHI_CASE_DIRECT_ROW_BLOCK_PRODUCT_SIMD_ALONG_M(9);
        MOCHI_CASE_DIRECT_ROW_BLOCK_PRODUCT_SIMD_ALONG_M(10);
        MOCHI_CASE_DIRECT_ROW_BLOCK_PRODUCT_SIMD_ALONG_M(11);
        MOCHI_CASE_DIRECT_ROW_BLOCK_PRODUCT_SIMD_ALONG_M(12);
        MOCHI_CASE_DIRECT_ROW_BLOCK_PRODUCT_SIMD_ALONG_M(13);
        MOCHI_CASE_DIRECT_ROW_BLOCK_PRODUCT_SIMD_ALONG_M(14);
        MOCHI_CASE_DIRECT_ROW_BLOCK_PRODUCT_SIMD_ALONG_M(15);
        MOCHI_UNLIKELY default : {
          static_assert(
              kRowsPerBatchHalf == 16, "Please update the cases in this switch statement");
          MOCHI_ASSERT_VERBOSE(false, "Unsupported number of leftover rows.");
        }
      }
    }
  } else {
    static_assert(mATC >= 0 && kLeftoverRows == 0, "Unsupported case");
  }
}

#undef MOCHI_CASE_DIRECT_ROW_BLOCK_PRODUCT_SIMD_ALONG_M

/// @brief Matrix-matrix product without explicit SIMD instructions.
template <
    typename Scalar,
    typename AccessorC,
    typename AccessorA,
    typename AccessorB,
    int mATC,
    int nATC,
    int kATC,
    typename MultScalar>
MOCHI_ANY void SimpleProduct(
    AccessorC&& C,
    AccessorA const& A,
    AccessorB const& B,
    IntOrEmpty<mATC> m,
    IntOrEmpty<nATC> n,
    IntOrEmpty<kATC> k,
    Multiplier<MultScalar> factor) {
  using NonConstScalar = std::remove_const_t<Scalar>;
  for (int r = 0; r < m; r++) {
    for (int c = 0; c < n; c++) {
      NonConstScalar val = 0;
      for (int i = 0; i < k; i++) {
        val += A(r, i) * B(i, c);
      }
      C.Store(r, c, factor.Apply(val));
    }
  }
}

/// @brief Auxiliary kernel to compute rows [r0, r0 + kNumRows) of the output matrix C in
/// DirectProductSimdAlongK.
/// @remarks
/// - Requires a scalar type with SIMD support.
/// - Most performant for large k, A stored row-major, and B stored column-major.
template <
    int kNumRows,
    typename Scalar,
    typename AccessorC,
    typename AccessorA,
    typename AccessorB,
    int mATC,
    int nATC,
    int kATC,
    typename MultScalar>
MOCHI_ANY inline void DirectRowBlockProductSimdAlongK(
    AccessorC&& C,
    AccessorA const& A,
    AccessorB const& B,
    [[maybe_unused]] IntOrEmpty<mATC> const& m,
    IntOrEmpty<nATC> const& n,
    IntOrEmpty<kATC> const& k,
    Multiplier<MultScalar> const& factor,
    int r0) {
  using NonConstScalar = std::remove_const_t<Scalar>;
  using VType = Simd<NonConstScalar>;
  static_assert(kNumRows > 0, "Number of rows must be positive");
  static_assert(VType::kIsSupported, "Implementation requires a scalar type with SIMD support");
  constexpr auto kVecSize = VType::kSize;
  MOCHI_ASSERT_VERBOSE((r0 >= 0) && (r0 + kNumRows <= m), "Out-of-range rows.");

  for (int c = 0; c < n; ++c) {
    int j = 0;
    VType s[kNumRows] = {};
    if constexpr (kATC == krylov::kDynamic || kATC >= kVecSize) {
      for (; j + kVecSize <= k; j += kVecSize) {
        for (int rr = 0, r = r0; rr < kNumRows; ++rr, ++r) {
          s[rr] += A.template RowVector<VType>(r, j) * B.template ColVector<VType>(j, c);
        }
      }
    }

    //--- Leftover entries. Note that {Row,Col}Vector<V, N>(r, c) and {Row,Col}Vector<V>(r, c, N)
    //--- return zeros for all entries after the N-th entry.
    constexpr int kLeftoverEntries = kATC % kVecSize;
    for (int rr = 0, r = r0; rr < kNumRows; ++rr, ++r) {
      if constexpr (kATC > 0 && kLeftoverEntries > 0) {
        MOCHI_ASSERT_VERBOSE(
            kLeftoverEntries == k.iVal() - j, "Inconsistent number of leftover entries.");
        s[rr] += A.template RowVector<VType, kLeftoverEntries>(r, j) *
            B.template ColVector<VType, kLeftoverEntries>(j, c);
      } else if constexpr (kATC == krylov::kDynamic) {
        int const leftoverEntries = k.iVal() - j;
        MOCHI_ASSERT_VERBOSE(
            leftoverEntries >= 0 && leftoverEntries < kVecSize,
            "Inconsistent number of leftover entries.");
        if (leftoverEntries > 0) {
          s[rr] += A.template RowVector<VType>(r, j, leftoverEntries) *
              B.template ColVector<VType>(j, c, leftoverEntries);
        }
      } else {
        static_assert(kATC >= 0 && kLeftoverEntries == 0, "Unsupported case");
      }
    }

    for (int rr = 0, r = r0; rr < kNumRows; ++rr, ++r) {
      if constexpr (std::is_same_v<MultScalar, void>) {
        C.Store(r, c, HSum(s[rr]));
      } else {
        C.Store(r, c, HSum(s[rr]) * Scalar(factor.factor));
      }
    }
  }
}

#define MOCHI_CASE_DIRECT_ROW_BLOCK_PRODUCT_SIMD_ALONG_K(NUM_ROWS)                  \
  case NUM_ROWS: {                                                                  \
    DirectRowBlockProductSimdAlongK<NUM_ROWS, Scalar>(C, A, B, m, n, k, factor, r); \
    break;                                                                          \
  }

/// @brief Matrix-matrix product using SIMD instructions along the "k" direction.
/// @remarks
/// - Requires a scalar type with SIMD support.
/// - Most performant for large k, A stored row-major, and B stored column-major.
template <
    typename Scalar,
    typename AccessorC,
    typename AccessorA,
    typename AccessorB,
    int mATC,
    int nATC,
    int kATC,
    typename MultScalar>
MOCHI_ANY inline void DirectProductSimdAlongK(
    AccessorC&& C,
    AccessorA const& A,
    AccessorB const& B,
    IntOrEmpty<mATC> const& m,
    IntOrEmpty<nATC> const& n,
    IntOrEmpty<kATC> const& k,
    Multiplier<MultScalar> const& factor) {
  //--- Compute the product in batches of rows to increase the number of concurrent FMA instructions
  //--- and reduce the performance hit due to latency. The optimal number of rows per batch depends
  //--- on the maximum number of concurrent FMA instructions and the number of registers on the
  //--- architecture. Empirical optimal value is consistently 4 or 8 on various x64 and ARM
  //--- architectures.
  constexpr int kRowsPerBatch = 8;
  int r = 0;
  if constexpr (mATC == krylov::kDynamic || mATC >= kRowsPerBatch) {
    for (; r + kRowsPerBatch <= m; r += kRowsPerBatch) {
      DirectRowBlockProductSimdAlongK<kRowsPerBatch, Scalar>(C, A, B, m, n, k, factor, r);
    }
  }
  //--- Leftover rows.
  constexpr int kLeftoverRows = mATC % kRowsPerBatch;
  if constexpr (mATC > 0 && kLeftoverRows > 0) {
    static_assert(kLeftoverRows < kRowsPerBatch, "Inconsistent number of leftover rows");
    DirectRowBlockProductSimdAlongK<kLeftoverRows, Scalar>(C, A, B, m, n, k, factor, r);
  } else if constexpr (mATC == krylov::kDynamic) {
    int const leftOverRows = m.iVal() - r;
    MOCHI_ASSERT_VERBOSE(
        leftOverRows >= 0 && leftOverRows < kRowsPerBatch, "Inconsistent number of leftover rows.");
    if (leftOverRows > 0) {
      //--- Performance note: Passing the number of leftover rows as runtime argument and removing
      //--- the switch statement degrades performance.
      switch (leftOverRows) {
        MOCHI_CASE_DIRECT_ROW_BLOCK_PRODUCT_SIMD_ALONG_K(1);
        MOCHI_CASE_DIRECT_ROW_BLOCK_PRODUCT_SIMD_ALONG_K(2);
        MOCHI_CASE_DIRECT_ROW_BLOCK_PRODUCT_SIMD_ALONG_K(3);
        MOCHI_CASE_DIRECT_ROW_BLOCK_PRODUCT_SIMD_ALONG_K(4);
        MOCHI_CASE_DIRECT_ROW_BLOCK_PRODUCT_SIMD_ALONG_K(5);
        MOCHI_CASE_DIRECT_ROW_BLOCK_PRODUCT_SIMD_ALONG_K(6);
        MOCHI_CASE_DIRECT_ROW_BLOCK_PRODUCT_SIMD_ALONG_K(7);
        MOCHI_UNLIKELY default : {
          static_assert(kRowsPerBatch == 8, "Please update the cases in this switch statement");
          MOCHI_ASSERT_VERBOSE(false, "Unsupported number of leftover rows.");
        }
      }
    }
  } else {
    static_assert(mATC >= 0 && kLeftoverRows == 0, "Unsupported case");
  }
}

#undef MOCHI_CASE_DIRECT_ROW_BLOCK_PRODUCT_SIMD_ALONG_K

#if MOCHI_USE_MATH_ACCELERATION
/// @brief Matrix-matrix product using GEMM routine from a math acceleration library.
/// @remarks
/// - The purpose of this routine is to leverage instructions and/or hardware that are only
///   available through a math acceleration library, e.g. AMX instructions via macOS Accelerate.
/// - This routine must run in the calling thread. If the acceleration library multi-threads by
///   default, multi-threading should be disabled.
template <
    typename Scalar,
    typename AccessorC,
    typename AccessorA,
    typename AccessorB,
    int mATC,
    int nATC,
    int kATC,
    typename MultScalar>
void GemmProduct(
    void (*gemmFunc)(
        CBLAS_ORDER,
        CBLAS_TRANSPOSE,
        CBLAS_TRANSPOSE,
        int32_t m,
        int32_t n,
        int32_t k,
        Scalar,
        Scalar const*,
        int32_t,
        Scalar const*,
        int32_t,
        Scalar,
        Scalar*,
        int32_t),
    AccessorC& C,
    AccessorA const& A,
    AccessorB const& B,
    IntOrEmpty<mATC> m,
    IntOrEmpty<nATC> n,
    IntOrEmpty<kATC> k,
    Multiplier<MultScalar> factor) {
  constexpr bool kIsAColMajor = (AccessorA::RowColCosts().first == 1);
  constexpr bool kIsBColMajor = (AccessorB::RowColCosts().first == 1);
  constexpr bool kIsCColMajor = (AccessorC::RowColCosts().first == 1);
  Scalar alpha = Scalar(1), beta = {};
  if constexpr (!std::is_same_v<MultScalar, void>) {
    alpha = Scalar(factor.factor);
  }
  if constexpr (AccessorC::kOp == DestOp::Set) {
    beta = Scalar(0);
  } else if constexpr (AccessorC::kOp == DestOp::NegSet) {
    alpha *= Scalar(-1);
    beta = Scalar(0);
  } else if constexpr (AccessorC::kOp == DestOp::Add) {
    beta = Scalar(1);
  } else {
    static_assert(AccessorC::kOp == DestOp::Sub, "Unsupported DestOp");
    alpha *= Scalar(-1);
    beta = Scalar(1);
  }
  gemmFunc(
      kIsCColMajor ? CblasColMajor : CblasRowMajor,
      (kIsAColMajor == kIsCColMajor) ? CblasNoTrans : CblasTrans,
      (kIsBColMajor == kIsCColMajor) ? CblasNoTrans : CblasTrans,
      m.iVal(),
      n.iVal(),
      k.iVal(),
      alpha,
      A.Data(),
      kIsAColMajor ? A.ColStride() : A.RowStride(),
      B.Data(),
      kIsBColMajor ? B.ColStride() : B.RowStride(),
      beta,
      C.Data(),
      kIsCColMajor ? C.ColStride() : C.RowStride());
}
#endif

/// @brief Pack the data from A into aBuffer by columns of twice the vector size.
template <typename VType, typename AccessorA, int kRowCount>
MOCHI_ANY void PackColData(
    AccessorA const& A,
    int row,
    int m,
    int k,
    typename VType::Scalar (*aBuffer)[kRowCount]) {
  // TODO(T158480383): Introduce minimum SIMD size to favor the SIMD implementation.
  using Scalar = typename VType::Scalar;
  static_assert(kRowCount == 2 * VType::kSize);
  constexpr auto kCostsA = AccessorA::RowColCosts();
  constexpr bool kUseSimd = VType::kIsSupported && (kCostsA.first < kCostsA.second);
  if constexpr (kUseSimd) {
    constexpr int kVecSize = VType::kSize;
    if (row + kRowCount < m) {
      for (int c = 0; c < k; ++c) {
        auto s1 = A.template ColVector<VType>(row, c);
        auto s2 = A.template ColVector<VType>(row + kVecSize, c);
        Store(&aBuffer[c][0], s1);
        Store(&aBuffer[c][kVecSize], s2);
      }
      return;
    }
  }
  for (int c = 0; c < k; ++c) {
    int r = 0;
    for (; r < m - row && r < kRowCount; ++r) {
      aBuffer[c][r] = A(row + r, c);
    }
    for (; r < kRowCount; ++r) {
      aBuffer[c][r] = Scalar(0);
    }
  }
}

template <typename VType, typename AccessorC, typename MultScalar, int kRowCount>
MOCHI_ANY void WriteResult(
    typename VType::Scalar cRes[][kRowCount],
    int row,
    int col,
    int m,
    int validCols,
    AccessorC& C,
    Multiplier<MultScalar> factor) {
  static_assert(VType::kIsSupported, "Implementation requires a scalar type with SIMD support");
  static_assert(kRowCount == 2 * VType::kSize);
  constexpr int kVecSize = VType::kSize;
  if (row + kRowCount <= m) {
    for (int c = 0; c < validCols; ++c) {
      auto s1 = Load<VType>(&cRes[c][0]);
      auto s2 = Load<VType>(&cRes[c][kVecSize]);
      C.StoreColVector(row, col + c, factor.Apply(s1));
      C.StoreColVector(row + kVecSize, col + c, factor.Apply(s2));
    }
  } else {
    for (int c = 0; c < validCols; ++c) {
      for (int r = 0; r < kRowCount && row + r < m; ++r) {
        C.Store(row + r, col + c, factor.Apply(cRes[c][r]));
      }
    }
  }
}

/// @brief Auxiliary kernel to compute rows [row0, row0 + kNumRows) of the output matrix C in
/// LargeProductPostTranspose, where kNumRows is twice the size of VType.
template <
    typename VType,
    typename AccessorC,
    typename AccessorA,
    typename AccessorB,
    int mATC,
    int nATC,
    int kATC,
    typename MultScalar,
    int kNumRows>
MOCHI_ANY MOCHI_FORCE_INLINE void RowBlockLargeProductPostTranspose(
    AccessorC&& C,
    AccessorA const& A,
    AccessorB const& B,
    IntOrEmpty<mATC> m,
    IntOrEmpty<nATC> n,
    IntOrEmpty<kATC> k,
    Multiplier<MultScalar> factor,
    int row0,
    typename VType::Scalar (*aBuffer)[kNumRows],
    typename VType::Scalar (*bBuffer)[6],
    typename VType::Scalar (&cBuffer)[6][kNumRows]) {
  static_assert(kNumRows == 2 * VType::kSize, "Number of rows must be twice the SIMD vector size.");
  // Pack the data from A(row0:row0 + kNumRows, 1:k) to aBuffer of size "kNumRows x k".
  PackColData<VType>(A, row0, m.iVal(), k.iVal(), &aBuffer[0]);
  for (int col = 0; col < n; col += 6) {
    int validCols = std::min(n.iVal() - col, 6);
    if (IsMemoryAccessor<AccessorB> && validCols == 6) {
      if constexpr (IsMemoryAccessor<AccessorB>) { // Protect from illegal Row/ColStride()
        MatMatKernel<VType>(
            &aBuffer[0][0], &B(0, col), &cBuffer[0][0], k.iVal(), B.RowStride(), B.ColStride());
      }
    } else {
      for (int c = 0; c < validCols; ++c) {
        for (int kk = 0; kk < k; ++kk) {
          bBuffer[kk][c] = B(kk, col + c);
        }
      }
      MatMatKernel<VType>(&aBuffer[0][0], &bBuffer[0][0], &cBuffer[0][0], k.iVal(), 6, 1);
    }
    WriteResult<VType>(cBuffer, row0, col, m.iVal(), validCols, C, factor);
  }
}

/// @brief Matrix-matrix product for large matrices.
/// @remarks
/// - Requires a scalar type with SIMD support.
/// - Optimized for large matrices with B stored column-major.
template <
    typename Scalar,
    typename AccessorC,
    typename AccessorA,
    typename AccessorB,
    int mATC,
    int nATC,
    int kATC,
    typename MultScalar>
MOCHI_ANY void LargeProductPostTranspose(
    AccessorC&& C,
    AccessorA const& A,
    AccessorB const& B,
    IntOrEmpty<mATC> m,
    IntOrEmpty<nATC> n,
    IntOrEmpty<kATC> k,
    Multiplier<MultScalar> factor) {
  //--- MatMatKernel is optimized for VType such that 16 VType's fill the floating-point SIMD
  //--- registers.
  using VType = Simd<Scalar, std::max(1, MOCHI_SIMD_REGISTER_COUNT / 16) * Simd<Scalar>::kSize>;
  using VTypeHalf = Simd<Scalar, VType::kSize / 2>;
  static_assert(
      VType::kIsSupported && VTypeHalf::kIsSupported,
      "Implementation requires a scalar type with SIMD support");
  constexpr int kRowsPerBatch = 2 * VType::kSize;
  constexpr int kRowsPerBatchHalf = kRowsPerBatch / 2;
  //--- Allocate memory for temporary buffer. Alignment is not required but it can reduce the number
  //--- of cache lines. Stack memory is used for sizes k < 200.
  constexpr size_t kAlignment = alignof(VType);
  constexpr auto getBufferSize = [=](size_t k) -> size_t {
    return (kRowsPerBatch + 6) * k * sizeof(Scalar);
  };
  MOCHI_FILO_STACK_ALLOCATOR(alloc, kAlignment + getBufferSize(200));
  size_t const bufferSize = getBufferSize(k.sVal());
  char* ptrBuffer = reinterpret_cast<char*>(alloc.allocate(bufferSize, kAlignment));
  auto* aBuffer = reinterpret_cast<Scalar(*)[kRowsPerBatch]>(ptrBuffer);
  auto* bBuffer =
      reinterpret_cast<Scalar(*)[6]>(ptrBuffer + kRowsPerBatch * k.sVal() * sizeof(Scalar));
  int const colOverflow = n.sVal() % 6;

  MOCHI_WARNING_PUSH();
  MOCHI_WARNING_IGNORE_MSVC(4127); // conditional expression is constant
  if (colOverflow != 0) {
    for (int c = colOverflow; c < 6; ++c) {
      for (int kk = 0; kk < k; ++kk) {
        bBuffer[kk][c] = Scalar(0);
      }
    }
  }
  MOCHI_WARNING_POP();

  alignas(kAlignment) Scalar cBuffer[6][kRowsPerBatch] MOCHI_NO_INIT;

  //--- Perform the product in batches of 'kRowsPerBatch' rows at a time, except the last batch that
  //--- uses half the rows (if possible) to improve performance.
  int row = 0;
  for (; row + kRowsPerBatch <= m; row += kRowsPerBatch) {
    RowBlockLargeProductPostTranspose<VType>(
        C, A, B, m, n, k, factor, row, aBuffer, bBuffer, cBuffer);
  }
  if ((row < m) && (m <= row + kRowsPerBatchHalf)) {
    //--- Only need to compute 'kRowsPerBatchHalf' rows.
    auto* aBufferHalf = reinterpret_cast<Scalar(*)[kRowsPerBatchHalf]>(ptrBuffer);
    auto* cBufferHalf = reinterpret_cast<Scalar(*)[6][kRowsPerBatchHalf]>(&cBuffer);
    RowBlockLargeProductPostTranspose<VTypeHalf>(
        C, A, B, m, n, k, factor, row, aBufferHalf, bBuffer, *cBufferHalf);
  } else if (row < m) {
    RowBlockLargeProductPostTranspose<VType>(
        C, A, B, m, n, k, factor, row, aBuffer, bBuffer, cBuffer);
  }

  //--- Deallocate temporary buffer.
  alloc.deallocate(ptrBuffer, bufferSize, kAlignment);
}

/// @brief Driver routine for the matrix-matrix product of large matrices.
/// @remarks
/// - Requires a scalar type with SIMD support.
/// - Determines if it's faster to compute the requested product or its transpose, and gates the
///   computation to the large product kernels.
template <
    typename Scalar,
    typename AccessorC,
    typename AccessorA,
    typename AccessorB,
    int mATC,
    int nATC,
    int kATC,
    typename MultScalar>
MOCHI_ANY void LargeProduct(
    AccessorC& C,
    AccessorA const& A,
    AccessorB const& B,
    IntOrEmpty<mATC> m,
    IntOrEmpty<nATC> n,
    IntOrEmpty<kATC> k,
    Multiplier<MultScalar> factor) {
#if MOCHI_USE_MATH_ACCELERATION

  // Compiler complains that cblas_dgemm is deprecated. There is a newer alternative, but upgrading
  // is not a priority at this time.
  MOCHI_WARNING_PUSH()
  MOCHI_WARNING_IGNORE_GCC_CLANG(GCC diagnostic ignored "-Wdeprecated-declarations")

  // Use accelerator if the number of FLOPs is >200k. The optimal threshold depends on the
  // acceleration library, hardware and matrix shape, among others. 200k is larger-than-optimal in
  // some cases but prevents from accelerating rectangular, moderate-size matrices for which the
  // accelerator is substantially slower than our implementation.
  if (int64_t(m.iVal()) * int64_t(n.iVal()) * int64_t(k.iVal()) >= 100000) {
    if constexpr (std::is_same_v<float const, Scalar const>) {
      return GemmProduct(cblas_sgemm, C, A, B, m, n, k, factor);
    } else if constexpr (std::is_same_v<double const, Scalar const>) {
      return GemmProduct(cblas_dgemm, C, A, B, m, n, k, factor);
    }
  }

  MOCHI_WARNING_POP()
#endif
  // LargeProductPostTranspose is optimized for column-major storage. If at least 2 of the matrices
  // are stored row-major, computing the transpose product is faster.
  constexpr auto kNumRowMajorMatrices = static_cast<int>(AccessorA::RowColCosts().second == 1) +
      static_cast<int>(AccessorB::RowColCosts().second == 1) +
      static_cast<int>(AccessorC::RowColCosts().second == 1);
  if constexpr (kNumRowMajorMatrices >= 2) {
    LargeProductPostTranspose<Scalar>(Transpose(C), Transpose(B), Transpose(A), n, m, k, factor);
  } else {
    LargeProductPostTranspose<Scalar>(C, A, B, m, n, k, factor);
  }
}

/// @brief Driver routine for the matrix-matrix product of small matrices and the matrix-vector
/// product of all matrices (independently of size).
/// @remarks
/// - Requires a scalar type with SIMD support.
/// - Determines
///   a) along which direction to use SIMD instructions
///   b) whether to compute the requested product or its transpose for best performance, and gates
///   the computation to the corresponding kernels.
template <
    typename Scalar,
    typename AccessorC,
    typename AccessorA,
    typename AccessorB,
    int mATC,
    int nATC,
    int kATC,
    typename MultScalar>
MOCHI_ANY void DirectProduct(
    AccessorC& C,
    AccessorA const& A,
    AccessorB const& B,
    IntOrEmpty<mATC> m,
    IntOrEmpty<nATC> n,
    IntOrEmpty<kATC> k,
    Multiplier<MultScalar> factor) {
  // TODO(T158480383): Introduce minimum SIMD size to favor the SIMD implementation.
  // TODO(T162131227): Introduce minimum matrix size to favor the SIMD implementation.
  using VType = Simd<std::remove_const_t<Scalar>>;
  static_assert(VType::kIsSupported, "Implementation requires a scalar type with SIMD support");
  constexpr auto kCostsA = AccessorA::RowColCosts();
  constexpr auto kCostsB = AccessorB::RowColCosts();
  if constexpr (kCostsA.first == 1 && kCostsB.first == 1) {
    // Col-Major x Col-Major, including Col-Major x Vector. Notes:
    // - Using SIMD instructions along m is almost always faster.
    // - Exceptions to this rule are "m < SIMD size" and "m << n and/or m << k". These cases could
    //   be optimized further.
    DirectProductSimdAlongM<Scalar>(C, A, B, m, n, k, factor);
  } else if constexpr (kCostsA.first == 1 && kCostsB.second == 1) {
    // Col-Major x Row-Major. Notes:
    // - Using SIMD instructions along m is almost always faster.
    // - The transpose product is computed if that improves utilization of SIMD instructions. This
    //   is an important optimization for triangular solve kernels that perform products of the form
    //   'a * B', where 'B' is a row-major matrix and 'a' is a row view of a col-major matrix.
    if constexpr (mATC >= 0 && nATC >= 0 && kATC >= 0) {
      if constexpr (mATC >= nATC || mATC >= VType::kSize / 2) {
        DirectProductSimdAlongM<Scalar>(C, A, B, m, n, k, factor);
      } else {
        DirectProductSimdAlongM<Scalar>(Transpose(C), Transpose(B), Transpose(A), n, m, k, factor);
      }
    } else if (m.iVal() >= n.iVal() || m.iVal() >= VType::kSize / 2) {
      DirectProductSimdAlongM<Scalar>(C, A, B, m, n, k, factor);
    } else {
      DirectProductSimdAlongM<Scalar>(Transpose(C), Transpose(B), Transpose(A), n, m, k, factor);
    }
  } else if constexpr (kCostsA.second == 1 && kCostsB.second == 1) {
    // Row-Major x Row-Major. Notes:
    // - Using SIMD instructions along m on the transpose product is almost always faster.
    // - Using SIMD instructions along k is faster if "n < SIMD size" and "n << k". This case is
    //   handled separately.
    // - On ARM, using SIMD instructions along k may also be faster if "k >> SIMD size" even if "n
    //   >= SIMD size", particularly for compile-time size matrices. For simplicity, this case isn't
    //   handled separately since it's rare and often gated through LargeProduct.
    // - The case "n < SIMD size" and "n, k << m" could also be optimized further.
    // NOLINTBEGIN(bugprone-branch-clone) - One branch is "if constexpr", the other can't be.
    if constexpr (mATC > 0 && nATC > 0 && kATC > 0) {
      DirectProductSimdAlongM<Scalar>(Transpose(C), Transpose(B), Transpose(A), n, m, k, factor);
    } else if (n.iVal() >= VType::kSize || k.iVal() < 4 * VType::kSize) {
      DirectProductSimdAlongM<Scalar>(Transpose(C), Transpose(B), Transpose(A), n, m, k, factor);
    } else {
      // Case "n < SIMD size" and "n << k".
      DirectProductSimdAlongK<Scalar>(C, A, B, m, n, k, factor);
    }
    // NOLINTEND(bugprone-branch-clone)
  } else if constexpr (kCostsA.second == 1 && kCostsB.first == 1) {
    // Row-Major x Col-Major, including Row-Major x Vector. Notes:
    // - Using SIMD instructions along k is almost always faster.
    // - For compile-time size matrices with k smaller than around the SIMD size, using SIMD
    //   instructions along m is usually faster, particularly on x64. This case is handled
    //   separately.
    // - The case "k < SIMD size" and "m, k << n" could be optimized further by using SIMD
    //   instructions along m on the transpose product.
    if constexpr (kATC == krylov::kDynamic || kATC >= VType::kSize) {
      DirectProductSimdAlongK<Scalar>(C, A, B, m, n, k, factor);
    } else {
      DirectProductSimdAlongM<Scalar>(C, A, B, m, n, k, factor);
    }
  } else {
    SimpleProduct<Scalar>(C, A, B, m, n, k, factor);
  }
}

/// @brief Driver routine for matrix-matrix and matrix-vector products.
/// @remarks
/// - Gates the product through NonSimdProduct (if the scalar type doesn't have SIMD support),
///   DirectProduct (if the scalar type has SIMD support and the matrices are small) or LargeProduct
///   (if the scalar type has SIMD support and the matrices are large).
/// - None of the kernels have been optimized for asymptotically large matrices. For asymptotically
///   large matrices, the product should be decomposed into smaller subproducts to improve cache
///   utilization.
template <
    typename Scalar,
    typename AccessorC,
    typename AccessorA,
    typename AccessorB,
    int mATC,
    int nATC,
    int kATC,
    typename MultScalar>
MOCHI_ANY void DoProduct(
    AccessorC& C,
    AccessorA const& A,
    AccessorB const& B,
    IntOrEmpty<mATC> m,
    IntOrEmpty<nATC> n,
    IntOrEmpty<kATC> k,
    Multiplier<MultScalar> factor) {
  MOCHI_ASSERT_VERBOSE(
      (m.iVal() * n.iVal() * k.iVal() == 0) || (C.Data() != A.Data() && C.Data() != B.Data()),
      "In-place product is not supported.");
  using VType = Simd<std::remove_const_t<Scalar>>;
  if constexpr (VType::kIsSupported && !MOCHI_ARCH_GPU) {
    // LargeProduct computes C in sub-blocks of size (m, n, k) = (N, 6, k), where N is a multiple of
    // the native SIMD vector size, and is in general faster than DirectProduct provided (m, n, k)
    // are sufficiently large. The optimal (m, n, k) thresholds to use LargeProduct depend on the
    // machine, scalar type, and major directions. The thresholds below have been tuned and are near
    // optimal on various x64 and ARM architectures. Notes:
    // - The thresholds have been tuned for single-precision arithmetic.
    // - The optimal thresholds for fully compile-time size matrices are in general slightly larger
    //   than the current values.
    // - The optimal thresholds for Row-Major x Col-Major are in general slightly larger than the
    //   current values.
    // - If performance is critical for some specific sizes and/or storage directions, the gating
    //   could be tuned further for them.
    // - See Summary of D52961265 for additional performance details.
    constexpr int kThresholdM = 16;
    constexpr int kThresholdN = 6;
    constexpr int kThresholdK = 16;
    if constexpr (mATC > 0 && nATC > 0 && kATC > 0) {
      // Fully compile-time size matrices.
      if constexpr (mATC < kThresholdM || nATC < kThresholdN || kATC < kThresholdK) {
        DirectProduct<Scalar>(C, A, B, m, n, k, factor);
      } else {
        LargeProduct<Scalar>(C, A, B, m, n, k, factor);
      }
    } else {
      // Fully dynamic or partially dynamic matrices.
      if (m.iVal() < kThresholdM || n.iVal() < kThresholdN || k.iVal() < kThresholdK) {
        DirectProduct<Scalar>(C, A, B, m, n, k, factor);
      } else {
        LargeProduct<Scalar>(C, A, B, m, n, k, factor);
      }
    }
  } else {
    SimpleProduct<Scalar>(C, A, B, m, n, k, factor);
  }
}

} // namespace mochi::details
