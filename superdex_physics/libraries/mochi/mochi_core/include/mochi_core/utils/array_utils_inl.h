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

// Reverse include for intellisense
#include "array_utils.h"

#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/nd_array_utils.h>
#include <mochi_core/utils/profile.h>

#include <algorithm>
#include <numeric>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mochi {

// NOTE: Many of the vertical operations in this header are simple enough that using scalar
// arithmetic and letting the compiler optimize them is faster than using explicit SIMD instructions
// (even with batching to hide latency), both with MSVC and Clang.

// Minimum number of values to compute per task for operations that are FAST (add, sub, etc...).
// This number is relatively large so that each task has a meaningful amount of work to do (shooting
// for at least 50 microseconds each)
int constexpr kMinPerTaskFast = 8 * 1024; // TODO: Current value is smaller than optimal.

template <typename T, typename SZ>
inline void ArrayPlusEquals(Span<T, SZ> dst, Span<T const, SZ> valuesToAddToDst) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT_VERBOSE(dst.size() == valuesToAddToDst.size(), "Inconsistent sizes.");
  ParallelForRange(
      "PlusEquals", 0, dst.size(), kMinPerTaskFast, INT_MAX, [&](int iBegin, int iEnd) {
        for (int i = iBegin; i < iEnd; ++i) {
          dst[i] += valuesToAddToDst[i];
        }
      });
}

template <typename T, typename SZ>
inline void ArrayMinusEquals(Span<T, SZ> dst, Span<T const, SZ> valuesToSubToDst) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT_VERBOSE(dst.size() == valuesToSubToDst.size(), "Inconsistent sizes.");
  ParallelForRange(
      "MinusEquals", 0, dst.size(), kMinPerTaskFast, INT_MAX, [&](int iBegin, int iEnd) {
        for (int i = iBegin; i < iEnd; ++i) {
          dst[i] -= valuesToSubToDst[i];
        }
      });
}

template <typename T, typename SZ>
inline void ArrayMulEquals(Span<T, SZ> dst, Span<T const, SZ> valuesToMulToDst) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT_VERBOSE(dst.size() == valuesToMulToDst.size(), "Inconsistent sizes.");
  ParallelForRange("MulEquals", 0, dst.size(), kMinPerTaskFast, INT_MAX, [&](int iBegin, int iEnd) {
    for (int i = iBegin; i < iEnd; ++i) {
      dst[i] *= valuesToMulToDst[i];
    }
  });
}

template <typename T, typename SZ>
inline void ArrayMulEquals(Span<T, SZ> dst, T alpha) {
  MOCHI_PROFILE_SCOPE();
  ParallelForRange(
      "MulAlphaEquals", 0, dst.size(), kMinPerTaskFast, INT_MAX, [&](int iBegin, int iEnd) {
        for (int i = iBegin; i < iEnd; ++i) {
          dst[i] *= alpha;
        }
      });
}

template <typename T, typename SZ>
inline void ArrayInverts(Span<T, SZ> dst) {
  MOCHI_PROFILE_SCOPE();
  ParallelForRange("Inverts", 0, isize(dst), kMinPerTaskFast, INT_MAX, [&](int iBegin, int iEnd) {
    for (int i = iBegin; i < iEnd; ++i) {
      dst[i] = T(1) / dst[i];
    }
  });
}

template <typename T, typename SZ>
inline void ArrayAdd(Span<T, SZ> dst, Span<T const, SZ> a, Span<T const, SZ> b) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT_VERBOSE(dst.size() == a.size() && dst.size() == b.size(), "Inconsistent sizes.");
  ParallelForRange("Add", 0, isize(dst), kMinPerTaskFast, INT_MAX, [&](int iBegin, int iEnd) {
    for (int i = iBegin; i < iEnd; ++i) {
      dst[i] = a[i] + b[i];
    }
  });
}

template <typename T>
inline void ArrayAdd(Span<NdArray<T, 3>> dst, Span<NdArray<T, 3> const> a, NdArray<T, 3> const& b) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT(dst.size() == a.size());
  if (dst.size() == 0) {
    return;
  }
  // Instead of adding 3 values at a time, we add blocks of 12 values (4 NdArray<T, 3> packed into a
  // Simd<T,12>). For the right-hand size, we pack 4 copies of b into a Simd<T, 12>.
  using V = Simd<T, 12>;
  V vb = {b[0], b[1], b[2], b[0], b[1], b[2], b[0], b[1], b[2], b[0], b[1], b[2]};
  int constexpr kCoordsPerBlock = 4;
  int const numBlocks = isize(dst) / kCoordsPerBlock;
  T* out = dst[0].data();
  T const* in = a[0].data();
  ParallelForN("Add", numBlocks, kMinPerTaskFast, [&](int i) {
    int offset = i * V::kSize;
    Store(out + offset, Load<V>(in + offset) + vb);
  });
  for (int i = numBlocks * kCoordsPerBlock; i < isize(dst); ++i) {
    dst[i] = a[i] + b;
  }
}

template <typename T, typename SZ>
inline void ArraySub(Span<T, SZ> dst, Span<T const, SZ> a, Span<T const, SZ> b) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT_VERBOSE(dst.size() == a.size() && dst.size() == b.size(), "Inconsistent sizes.");
  ParallelForRange("Sub", 0, dst.size(), kMinPerTaskFast, INT_MAX, [&](int iBegin, int iEnd) {
    for (int i = iBegin; i < iEnd; ++i) {
      dst[i] = a[i] - b[i];
    }
  });
}

template <int N, typename T>
inline void ArrayGetTriples(T const* src, Span<int const> const& indices, T* dst) {
  if constexpr (N != krylov::kDynamic) {
    int const* indicesData = indices.data();
    static_assert(N >= 0, "Requires a non-negative number of values");
    static_assert((N % 3) == 0, "Requires value triples");
#if MOCHI_ASSERT_VERBOSE_ENABLED
    for (int i = 0; i < N; i += 3) {
      MOCHI_ASSERT_VERBOSE(
          indicesData[i + 1] == indicesData[i] + 1, "Requires consecutive indices in groups of 3");
      MOCHI_ASSERT_VERBOSE(
          indicesData[i + 2] == indicesData[i] + 2, "Requires consecutive indices in groups of 3");
    }
#endif
    using V = Simd<T, 4>;
    if constexpr (N == 12) {
      // Special case for 3D tetrahedrons
      Store(dst + 0, Load<3, V>(src + indicesData[0]));
      Store(dst + 3, Load<3, V>(src + indicesData[3]));
      Store(dst + 6, Load<3, V>(src + indicesData[6]));
      Store<3>(dst + 9, Load<3, V>(src + indicesData[9]));
    } else {
      for (int i = 0; i < N; i += 3) {
        int const ii = indicesData[i];
        Store<3>(dst, Load<3, V>(src + ii));
        dst += 3;
      }
    }
  } else {
    int const n = isize(indices);
    MOCHI_ASSERT_VERBOSE((n % 3) == 0, "Requires value triples");
    for (int i = 0; i < n; i += 3) {
      Store<3>(dst, Load<3, Simd<T, 4>>(src + indices[i]));
      dst += 3;
    }
  }
}

template <typename T>
inline void ArrayAddTriples(T* vec, int n, int const* indices, T const* values) {
  MOCHI_ASSERT_VERBOSE((n % 3) == 0, "Requires a multiple of 3 indices and values");
  using V = Simd<T, 4>;
  for (int i = 0; i < n; i += 3, values += 3) {
    MOCHI_ASSERT_VERBOSE(
        indices[i + 1] == indices[i] + 1, "Requires consecutive indices in groups of 3");
    MOCHI_ASSERT_VERBOSE(
        indices[i + 2] == indices[i] + 2, "Requires consecutive indices in groups of 3");
    int const ii = indices[i];
    T* p = vec + ii;
    V a = Load<3, V>(p);
    V b = Load<3, V>(values);
    Store<3>(p, a + b);
  }
}

template <int N, typename T>
inline void ArrayAddTriplesN(T* vec, Span<int const> const& indices, T const* values) {
  if constexpr (N != krylov::kDynamic) {
    static_assert(N >= 0, "Requires a non-negative number of values");
    static_assert((N % 3) == 0, "Requires a multiple of 3 indices and values");
    MOCHI_ASSERT_VERBOSE(isize(indices) >= N, "Not enough indices provided");
    using V = Simd<T, 4>;
    if constexpr (N == 12) {
      // Special case for tetrahedral elements (4 verts per element, 3 values per vert)
      Store<3>(vec + indices[0], Load<3, V>(vec + indices[0]) + Load<3, V>(&values[0]));
      Store<3>(vec + indices[3], Load<3, V>(vec + indices[3]) + Load<3, V>(&values[3]));
      Store<3>(vec + indices[6], Load<3, V>(vec + indices[6]) + Load<3, V>(&values[6]));
      Store<3>(vec + indices[9], Load<3, V>(vec + indices[9]) + Load<3, V>(&values[9]));
    } else {
      ArrayAddTriples(vec, N, indices.data(), values);
    }
  } else {
    ArrayAddTriples(vec, isize(indices), indices.data(), values);
  }
}

// This overload takes a TRANSPOSED 4x4 matrix
template <bool kSingleThreaded = false, typename T>
inline void ArrayTransformPoints_MatT(
    Span<NdArray<T, 3>> dst,
    Span<NdArray<T, 3> const> src,
    NdArray<Simd<T, 4>, 4> const& matT) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT(dst.size() == src.size());
  int const count = isize(src);
  if (count == 0) {
    return;
  }
  auto forRange = [&](int iBegin, int iEnd) {
    using V = Simd<T>;
    // Benchmarking shows no benefit to larger-than-native batch sizes for both NEON and AVX2.
    int constexpr kBatchSize = Simd<T>::kSize;
    int i = iBegin;
    if (i + kBatchSize <= iEnd) {
      // Vectorize rotation and translation. Ignore the right column of the matrix.
      auto const rotT = Broadcast3x3<V>(matT); // 3x3xN
      auto const trans = Broadcast3<V>(matT[3]); // 3xN
      for (; i + kBatchSize <= iEnd; i += kBatchSize) {
        NdArray<V, 3> pt;
        LoadTransposed(&src[i][0], pt);
        auto result = DotVecMat(pt, rotT) + trans;
        StoreTransposed(&dst[i][0], result);
      }
    }
    for (; i < iEnd; ++i) {
      auto pt = Load<3, Simd<T, 4>>(&src[i][0]);
      auto result = DotVecMat4x4(ToSimdPoint(pt), matT);
      Store<3>(&dst[i][0], result);
    }
  };
  if constexpr (kSingleThreaded) {
    forRange(0, count);
  } else {
    int constexpr kMinPerTask = 8 * 1024;
    ParallelForRange("TransformPoints", 0, count, kMinPerTask, INT_MAX, forRange);
  }
}

template <bool kSingleThreaded, typename T>
inline void ArrayTransformPoints(
    Span<NdArray<T, 3>> dst,
    Span<NdArray<T, 3> const> src,
    TransformSRT const& transform) {
  using VMatrix4x4t = NdArray<Simd<T, 4>, 4>;
  ArrayTransformPoints_MatT<kSingleThreaded>(
      dst,
      src,
      StaticCast<VMatrix4x4t>(
          ToVMatrix4x4Transpose(transform))); // Cast scalars from real to T (if different)
}

// This overload takes a TRANSPOSED matrix
template <bool kSingleThreaded, typename T>
inline void ArrayTransformDisplacements_MatT(
    Span<NdArray<T, 3>> dstDisplacements,
    Span<NdArray<T, 3> const> srcDisplacements,
    Span<NdArray<T, 3> const> refCoords,
    NdArray<Simd<T, 4>, 4> const& matT) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT(dstDisplacements.size() == srcDisplacements.size());
  MOCHI_ASSERT(dstDisplacements.size() == refCoords.size());
  int const count = isize(refCoords);
  if (count == 0) {
    return;
  }
  auto forRange = [&](int iBegin, int iEnd) {
    using V = Simd<T>;
    // Benchmarking shows no benefit to larger-than-native batch sizes for both NEON and AVX2.
    int constexpr kBatchSize = Simd<T>::kSize;
    int i = iBegin;
    if (i + kBatchSize <= iEnd) {
      // Vectorize rotation and translation. Ignore the right column of the matrix.
      auto const rotT = Broadcast3x3<V>(matT); // 3x3xN
      auto const trans = Broadcast3<V>(matT[3]); // 3xN
      for (; i + kBatchSize <= iEnd; i += kBatchSize) {
        NdArray<V, 3> ref, pt;
        LoadTransposed(&refCoords[i][0], ref);
        LoadTransposed(&srcDisplacements[i][0], pt);
        auto result = DotVecMat(pt + ref, rotT) + trans - ref;
        StoreTransposed(&dstDisplacements[i][0], result);
      }
    }
    for (; i < iEnd; ++i) {
      auto ref = Load<3, Simd<T, 4>>(refCoords[i].data());
      auto pt = Load<3, Simd<T, 4>>(srcDisplacements[i].data());
      pt = DotVecMat4x4(ToSimdPoint(pt + ref), matT) - ref;
      Store<3>(dstDisplacements[i].data(), pt);
    }
  };
  if constexpr (kSingleThreaded) {
    forRange(0, count);
  } else {
    int constexpr minPerTask = 8 * 1024;
    ParallelForRange("TransformDisplacements", 0, count, minPerTask, INT_MAX, forRange);
  }
}

// This overload takes a TRANSPOSED matrix
template <bool kSingleThreaded, typename T>
inline void ArrayTransformDisplacements_MatT(
    ColumnVectorView<T> dstDisplacements,
    ColumnVectorView<T const> srcDisplacements,
    ColumnVectorView<T const> refCoordsFlat,
    NdArray<Simd<T, 4>, 4> const& matT) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT(
      (dstDisplacements.Rows() == srcDisplacements.Rows()) &&
          (dstDisplacements.Rows() == refCoordsFlat.Rows()),
      "Size mismatch");
  MOCHI_ASSERT(refCoordsFlat.Rows() % 3 == 0, "Expected 3D coordinates");
  T* dstData = dstDisplacements.data();
  T const* srcData = srcDisplacements.data();
  T const* refCoordsData = refCoordsFlat.data();
  ArrayTransformDisplacements_MatT<kSingleThreaded>(
      Unflatten<NdArray<T, 3>>(Span<T>{dstData, (size_t)dstDisplacements.Rows()}),
      Unflatten<NdArray<T, 3> const>(Span<T const>{srcData, (size_t)srcDisplacements.Rows()}),
      Unflatten<NdArray<T, 3> const>(Span<T const>{refCoordsData, (size_t)refCoordsFlat.Rows()}),
      matT);
}

template <bool kSingleThreaded, typename T>
inline void ArrayTransformDisplacements(
    Span<NdArray<T, 3>> dstDisplacements,
    Span<NdArray<T, 3> const> srcDisplacements,
    Span<NdArray<T, 3> const> refCoords,
    TransformRT const& transform) {
  using VMatrix4x4t = NdArray<Simd<T, 4>, 4>;
  ArrayTransformDisplacements_MatT<kSingleThreaded>(
      dstDisplacements,
      srcDisplacements,
      refCoords,
      StaticCast<VMatrix4x4t>(
          ToVMatrix4x4Transpose(transform))); // Cast scalars from real to T (if different)
}

// This overload takes a TRANSPOSED 3x3 rotation matrix
template <bool kSingleThreaded, typename T, size_t kNumRows>
  requires(kNumRows == 3 || kNumRows == 4)
void ArrayRotateVectors_MatT(
    Span<NdArray<T, 3>> dst,
    Span<NdArray<T, 3> const> src,
    NdArray<Simd<T, 4>, kNumRows> const& matT) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT(dst.size() == src.size());
  int const count = isize(src);
  if (count == 0) {
    return;
  }
  auto forRange = [&](int iBegin, int iEnd) {
    using V = Simd<T>;
    // Benchmarking shows no benefit to larger-than-native batch sizes for both NEON and AVX2.
    int constexpr kBatchSize = Simd<T>::kSize;
    int i = iBegin;
    if (i + kBatchSize <= iEnd) {
      // Vectorize rotation. Ignore the right column of the matrix.
      auto const rotT = Broadcast3x3<V>(matT); // 3x3xN
      for (; i + kBatchSize <= iEnd; i += kBatchSize) {
        NdArray<V, 3> pt;
        LoadTransposed(&src[i][0], pt);
        auto result = DotVecMat(pt, rotT);
        StoreTransposed(&dst[i][0], result);
      }
    }
    for (; i < iEnd; ++i) {
      auto pt = Load<3, Simd<T, 4>>(&src[i][0]);
      auto result = DotVecMat3x3(pt, matT);
      Store<3>(&dst[i][0], result);
    }
  };
  if constexpr (kSingleThreaded) {
    forRange(0, count);
  } else {
    int constexpr minPerTask = 8 * 1024;
    ParallelForRange("RotateVectors", 0, count, minPerTask, INT_MAX, forRange);
  }
}

// This overload takes a TRANSPOSED 3x3 rotation matrix
template <bool kSingleThreaded, typename T, size_t kNumRows>
  requires(kNumRows == 3 || kNumRows == 4)
void ArrayRotateVectors_MatT(
    ColumnVectorView<T> dst,
    ColumnVectorView<T const> src,
    NdArray<Simd<T, 4>, kNumRows> const& matT) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT(dst.Rows() == src.Rows());
  MOCHI_ASSERT(dst.Rows() % 3 == 0, "Expected 3D coordinates");
  T* dstData = dst.data();
  T const* srcData = src.data();
  ArrayRotateVectors_MatT<kSingleThreaded>(
      Unflatten<NdArray<T, 3>>(Span<T>{dstData, (size_t)dst.Rows()}),
      Unflatten<NdArray<T, 3> const>(Span<T const>{srcData, (size_t)src.Rows()}),
      matT);
}

template <bool kSingleThreaded, typename T>
inline void ArrayRotateVectors(
    Span<NdArray<T, 3>> dst,
    Span<NdArray<T, 3> const> src,
    Quaternion const& rotation) {
  using VMatrix3x3t = NdArray<Simd<T, 4>, 3>;
  return ArrayRotateVectors_MatT<kSingleThreaded>(
      dst,
      src,
      StaticCast<VMatrix3x3t>(
          ToVMatrix3x3Transpose(rotation))); // Cast scalars from real to T (if different)
}

template <typename T, typename SZ>
T Min(Span<T const, SZ> a) {
  MOCHI_ASSERT_VERBOSE(!a.empty(), "Cannot compute min of an empty array");
  auto const len = a.size();
  auto i = SZ(1);
  T min = a[0];
  using VecT = Simd<T>;
  if constexpr (VecT::kIsSupported) {
    if (len >= VecT::kSize) {
      VecT vmin = Load<VecT>(&a[0]);
      for (i = VecT::kSize; i + VecT::kSize <= len; i += VecT::kSize) {
        vmin = Min(vmin, Load<VecT>(&a[i]));
      }
      min = HMin(vmin);
    }
  }
  for (; i < len; ++i) {
    min = Min(min, a[i]);
  }
  return min;
}

template <typename T, typename SZ>
T Max(Span<T const, SZ> a) {
  MOCHI_ASSERT_VERBOSE(!a.empty(), "Cannot compute max of an empty array");
  auto const len = a.size();
  auto i = SZ(1);
  T max = a[0];
  using VecT = Simd<T>;
  if constexpr (VecT::kIsSupported) {
    if (len >= VecT::kSize) {
      VecT vmax = Load<VecT>(&a[0]);
      for (i = VecT::kSize; i + VecT::kSize <= len; i += VecT::kSize) {
        vmax = Max(vmax, Load<VecT>(&a[i]));
      }
      max = HMax(vmax);
    }
  }
  for (; i < len; ++i) {
    max = Max(max, a[i]);
  }
  return max;
}

template <typename T, typename SZ>
std::pair<T, T> MinMax(Span<T const, SZ> a) {
  MOCHI_ASSERT(!a.empty(), "Cannot compute min/max of an empty array");
  auto const len = a.size();
  auto i = SZ(1);
  auto min = a[0];
  auto max = min;
  // Start with SIMD blocks, if supported
  using VecT = Simd<T>;
  if (VecT::kIsSupported && (len >= VecT::kSize)) {
    VecT vmin = Load<VecT>(&a[0]);
    VecT vmax = vmin;
    for (i = VecT::kSize; i + VecT::kSize <= len; i += VecT::kSize) {
      VecT val = Load<VecT>(&a[i]);
      vmin = Min(vmin, val);
      vmax = Max(vmax, val);
    }
    min = HMin(vmin);
    max = HMax(vmax);
  }
  // Then compare any remaining items
  for (; i < len; ++i) {
    min = Min(min, a[i]);
    max = Max(max, a[i]);
  }
  return std::make_pair(min, max);
}

template <typename T, typename SZ>
inline T MaxAbs(Span<T const, SZ> a) {
  auto const len = a.size();
  auto max = T(0);
  auto i = SZ(0);
  // Start with SIMD blocks, if supported
  using VecT = Simd<std::remove_const_t<T>>;
  if constexpr (VecT::kIsSupported) {
    if (len >= VecT::kSize) {
      VecT vmax = Abs(Load<VecT>(&a[0]));
      for (i = VecT::kSize; i + VecT::kSize <= len; i += VecT::kSize) {
        vmax = Max(vmax, Abs(Load<VecT>(&a[i])));
      }
      max = HMax(vmax);
    }
  }
  // Then compare any remaining items
  for (; i < len; ++i) {
    max = Max(max, Abs(a[i]));
  }
  return max;
}

template <typename T, typename SZ>
inline T MaxAbsDifference(Span<T const, SZ> a, Span<T const, SZ> b) {
  MOCHI_ASSERT(a.size() == b.size());
  auto const len = a.size();
  auto i = SZ(0);
  auto maxDiff = T(0);
  // Start with SIMD blocks, if supported
  using VecT = Simd<std::remove_const_t<T>>;
  if constexpr (VecT::kIsSupported) {
    VecT vMaxDiff = {};
    for (; i + VecT::kSize <= len; i += VecT::kSize) {
      auto va = Load<VecT>(&a[i]);
      auto vb = Load<VecT>(&b[i]);
      vMaxDiff = Max(vMaxDiff, Abs(va - vb));
    }
    maxDiff = HMax(vMaxDiff);
  }
  // Then compare any remaining items
  for (; i < len; ++i) {
    maxDiff = Max(maxDiff, Abs(a[i] - b[i]));
  }
  return maxDiff;
}

template <typename T, typename SZ>
T HSum(Span<T, SZ> a) {
  auto hsum = T{};
  auto i = SZ(0);
  auto n = a.size();
  using VT = Simd<std::remove_const_t<T>>;
  if constexpr (VT::kIsSupported) {
    int constexpr kNumBlocks = 4; // Large enough to hide latency
    int constexpr kBlockStride = VT::kSize * kNumBlocks;
    VT block[kNumBlocks] = {};
    if (n >= kBlockStride) {
      for (; i + kBlockStride <= n; i += kBlockStride) {
        auto tmp0 = Load<VT>(&a[i + VT::kSize * 0]);
        auto tmp1 = Load<VT>(&a[i + VT::kSize * 1]);
        auto tmp2 = Load<VT>(&a[i + VT::kSize * 2]);
        auto tmp3 = Load<VT>(&a[i + VT::kSize * 3]);
        block[0] += tmp0;
        block[1] += tmp1;
        block[2] += tmp2;
        block[3] += tmp3;
      }
      block[0] += block[1] + block[2] + block[3];
    }
    for (; i + VT::kSize <= n; i += VT::kSize) {
      block[0] += Load<VT>(&a[i]);
    }
    hsum = HSum(block[0]);
  }
  for (; i < n; ++i) {
    hsum += a[i];
  }
  return hsum;
}

template <typename ContainerT>
void SortAndRemoveDuplicates(ContainerT& v) {
  bool const isSorted = std::is_sorted(v.begin(), v.end());
  if (!isSorted) {
    std::sort(v.begin(), v.end());
  }
  auto newEnd = std::unique(v.begin(), v.end());
  v.resize(static_cast<size_t>(newEnd - v.begin()));
}

template <typename T, typename SZ>
[[nodiscard]] bool IsUnique(Span<T const, SZ> v) {
  std::unordered_set<T> set;
  for (auto const& e : v) {
    if (!set.insert(e).second) {
      return false;
    }
  }
  return true;
}

template <typename IndexType, typename T>
[[nodiscard]] DynamicArray<IndexType> ArgSort(Span<T const> v) {
  DynamicArray<IndexType> indices;
  indices.resize_noinit(v.size());
  std::iota(indices.begin(), indices.end(), IndexType(0));
  std::sort(indices.begin(), indices.end(), [&](IndexType left, IndexType right) -> bool {
    return v[left] < v[right];
  });
  return indices;
}

template <class T, typename SizeT>
[[nodiscard]] bool IsFinite(Span<T, SizeT> const& a) {
  SizeT i = 0;
  bool isFinite = true;
  using VecT = Simd<std::remove_const_t<T>>;
  if constexpr (VecT::kIsSupported) {
    for (; i + VecT::kSize <= a.size(); i += VecT::kSize) {
      isFinite &= IsFinite(Load<VecT>(&a[i]));
    }
  }
  for (; i < a.size(); ++i) {
    isFinite &= IsFinite(a[i]);
  }
  return isFinite;
}

template <class T>
void Fill(Span<T> dst, T value) {
  if constexpr (
      std::is_class_v<T> && std::is_trivially_copyable_v<T> && Simd<int>::kIsSupported &&
      (sizeof(T) % sizeof(int) == 0)) {
    // Batch size in bytes must be lowest common multiple of value size and SIMD byte size
    constexpr size_t kBatchSizeBytes =
        (sizeof(T) * sizeof(Simd<int>)) / std::gcd(sizeof(T), sizeof(Simd<int>));
    static_assert(kBatchSizeBytes % sizeof(T) == 0); // just checking
    static_assert(kBatchSizeBytes % sizeof(int) == 0); // just checking
    constexpr size_t kBatchSize = kBatchSizeBytes / sizeof(T);
    constexpr size_t kSimdIntCount = kBatchSizeBytes / sizeof(int);

    size_t const count = dst.size();
    size_t i = 0;

    if (count >= kBatchSize) {
      using V = Simd<int, kSimdIntCount>;
      // Create a buffer with kBatchSize copies of the value
      alignas(Max(alignof(V), alignof(T))) T srcBuf[kBatchSize];
      for (size_t j = 0; j < kBatchSize; ++j) {
        srcBuf[j] = value;
      }
      // Load the buffer into Simd
      auto src = Load<V>(reinterpret_cast<int const*>(srcBuf));
      // Copy batches to the destination
      for (; i + kBatchSize <= count; i += kBatchSize) {
        Store(reinterpret_cast<int*>(&dst[i]), src);
      }
    }

    // Fill remaining elements
    for (; i < count; ++i) {
      dst[i] = value;
    }
  } else {
    // Fallback for all other types
    std::ranges::fill(dst, value);
  }
}

// Specialized for copying NdArray<T, N> (e.g. Real3). Even faster than the generic version above.
template <class T, size_t N>
  requires(Simd<T>::kIsSupported)
void Fill(Span<NdArray<T, N>> dst, NdArray<T, N> value) {
  constexpr size_t kBatchSize = Simd<T>::kSize;
  size_t const count = dst.size();
  size_t i = 0;
  if (dst.size() >= kBatchSize) {
    // Initialize a Simd vector with kBatchSize copies of the input value
    alignas(Simd<T>) NdArray<T, N> srcBuf[kBatchSize];
    for (size_t j = 0; j < kBatchSize; ++j) {
      srcBuf[j] = value;
    }
    auto src = Load<Simd<T, kBatchSize * N>>(&srcBuf[0][0]);
    // Copy batches to the destination
    for (; i + kBatchSize <= count; i += kBatchSize) {
      Store(&dst[i][0], src);
    }
  }
  for (; i < count; ++i) {
    dst[i] = value;
  }
}

} // namespace mochi
