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

#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/span.h>

namespace mochi {

// Common SIMD array types.

// NOTE: While most of these are redundant (multiple matrix
// shapes fallback to the same underlying NdArray type),
// they are mostly used to convey intention and expected
// data in the functions using them.

/**
 Special case for the SIMD representation of a 2x2 matrix.
 Stores the matrix entries following a flattened row-major
 layout. That is, in the order (a, c, d, b), corresponding
 to the matrix:
  | a c |
  | d b |
 */

using VMatrix2x2r = Simd<real, 4>;
using VMatrix2x2f = Simd<float, 4>;

/**
 SIMD representation of a NxM matrices. Stores the matrix entries in N vectors following the
 standard-row major layout, with as many padding entries as necessary (ok, only either 0 or 1).
 */

using VMatrix3x2r = NdArray<Simd<real, 4>, 3>;
using VMatrix2x3r = NdArray<Simd<real, 4>, 2>;
using VMatrix2x4r = NdArray<Simd<real, 4>, 2>;
using VMatrix3x3r = NdArray<Simd<real, 4>, 3>;
using VMatrix3x4r = NdArray<Simd<real, 4>, 3>;
using VMatrix4x3r = NdArray<Simd<real, 4>, 4>;
using VMatrix4x4r = NdArray<Simd<real, 4>, 4>;

using VMatrix3x2f = NdArray<Simd<float, 4>, 3>;
using VMatrix2x3f = NdArray<Simd<float, 4>, 2>;
using VMatrix2x4f = NdArray<Simd<float, 4>, 2>;
using VMatrix3x3f = NdArray<Simd<float, 4>, 3>;
using VMatrix3x4f = NdArray<Simd<float, 4>, 3>;
using VMatrix4x3f = NdArray<Simd<float, 4>, 4>;
using VMatrix4x4f = NdArray<Simd<float, 4>, 4>;

using VMatrix3x3d = NdArray<Simd<double, 4>, 3>;

/**
 SIMD representation of tensors.
 */
using VTensor3x3x3x3r = NdArray<Simd<real, 4>, 3, 3, 3>;
using VTensor3x3x3r = NdArray<Simd<real, 4>, 3, 3>;
using VTensor4x3x3r = NdArray<Simd<real, 4>, 4, 3>;

using VTensor3x3x3x3f = NdArray<Simd<float, 4>, 3, 3, 3>;
using VTensor3x3x3f = NdArray<Simd<float, 4>, 3, 3>;

/**
 Special case for the SIMD representation of a 2x2 symmetric
 matrix. Stores the matrix entries in the order (a, c, b, ?),
 corresponding to the symmetric matrix:
  | a c |
  | c b |
 */
using VSymMatrix2x2r = Simd<real, 4>;
using VSymMatrix2x2f = Simd<float, 4>;

/**
 Special case for the SIMD representation of a 3x3 symmetric
 matrix. Stores the matrix entries in two vectors, containing
 the diagonal terms (a, b, c, ?) and the off-diagonal terms
 (d, e, f, ?). Corresponds to the symmetric matrix:
  | a d e |
  | d b f |
  | e f c |
 */
using VSymMatrix3x3r = NdArray<Simd<real, 4>, 2>;
using VSymMatrix3x3f = NdArray<Simd<float, 4>, 2>;

// Vec4r full matrix <--> Vec4r sym matrix conversion
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 3> SimdSymToFull(
    NdArray<Simd<T, 4>, 2> const& m);
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE Simd<T, 4> SimdSymToFull(Simd<T, 4> const& m);
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 2> SimdFullToSym(
    NdArray<Simd<T, 4>, 3> const& m);
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE Simd<T, 4> SimdFullToSym(Simd<T, 4> const& m);

// NdArray <--> Vec4r conversion
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE Real2 ToReal2(Simd<T, 4> v);
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE Real3 ToReal3(Simd<T, 4> v);
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE Real4 ToReal4(Simd<T, 4> v);
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<T, 2, 2> ToNdArray2x2(Simd<T, 4> m);
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<T, 3, 3> ToNdArray3x3(NdArray<Simd<T, 4>, 3> m);
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<T, 4, 4> ToNdArray(NdArray<Simd<T, 4>, 4> m);
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<T, 2, 2> ToNdArraySym2x2(Simd<T, 4> m);
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<T, 3, 3> ToNdArraySym3x3(NdArray<Simd<T, 4>, 2> m);
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE Simd<T, 4> ToSimd(NdArray<T, 2> const& v, T z = T(0), T w = T(0));
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE Simd<T, 4> ToSimd(NdArray<T, 3> const& v, T w = T(0));
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE Simd<T, 4> ToSimd(NdArray<T, 4> v);
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE Simd<T, 4> ToSimdMatrix(NdArray<T, 2, 2> const& m);
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 3> ToSimdMatrix(NdArray<T, 3, 3> const& m);
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 4> ToSimdMatrix(NdArray<T, 4, 4> const& m);
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE Simd<T, 4> ToSimdSymMatrix(NdArray<T, 2, 2> const& m);
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 2> ToSimdSymMatrix(NdArray<T, 3, 3> const& m);
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<T, 3, 3, 3, 3> ToNdArrayTensor(
    NdArray<Simd<T, 4>, 3, 3, 3> const& t);
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 3, 3, 3> ToSimdTensor(
    NdArray<T, 3, 3, 3, 3> const& t);
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE Simd<T, 4> ToSimdFromSymComponents2x2(T a00, T a01, T a11);

/**************************************************************************************************
  NdArray conversions for simd <--> scalar

  If the last dimension != 4, then the Simd<T, 4> representation will have padding.
  The value of the padding could be anything. Don't assume zero.
*/

// Load a dense (D0 x D1) row-major matrix into an array of Simd<T, 4> for SIMD operations.
template <size_t D0, size_t D1, typename T>
MOCHI_FORCE_INLINE void LoadMatrix(NdArray<Simd<T, 4>, D0>& out, NdArray<T, D0, D1> const& x);

// Load a dense (D0 x D1) row-major matrix into an array of Simd<T, 4> for SIMD operations.
// For this overload, the dense matrix is represented as flat Span of (D0 * D1) values.
template <size_t D0, size_t D1, typename T, typename ConstOrNonConstT>
MOCHI_FORCE_INLINE void LoadMatrix(NdArray<Simd<T, 4>, D0>& out, Span<ConstOrNonConstT> x);

// Load a dense (D0 x D1) row-major matrix into an array of Simd<T, 4> for SIMD operations.
// For this overload, the dense matrix is represented as flat pointer.
template <size_t D0, size_t D1, typename T>
MOCHI_FORCE_INLINE void LoadMatrix(NdArray<Simd<T, 4>, D0>& out, T const* x);

// Load a dense 4D row-major matrix into a 3D array of Simd<T, 4> for SIMD operations.
template <size_t D0, size_t D1, size_t D2, size_t D3, typename T>
MOCHI_FORCE_INLINE void LoadMatrix(
    NdArray<Simd<T, 4>, D0, D1, D2>& out,
    NdArray<T, D0, D1, D2, D3> const& x);

// Load a dense 4D row-major matrix into a 3D array of Simd<T, 4> for SIMD operations.
// For this overload, the dense matrix is represented as flat pointer.
template <size_t D0, size_t D1, size_t D2, size_t D3, typename T>
MOCHI_FORCE_INLINE void LoadMatrix(NdArray<Simd<T, 4>, D0, D1, D2>& out, T const* x);

// Load a dense (D0 x D1) row-major submatrix using an array of Simd<T, 4> (D1 <= 4, skipping
// padding if D1 != 4) from a dense (D2 x D3) row-major matrix at the the specified row and column.
template <size_t D0, size_t D1, size_t D2, size_t D3, typename T>
MOCHI_FORCE_INLINE void
LoadSubmatrix(NdArray<Simd<T, 4>, D0>& out, Int2 coords, NdArray<T, D2, D3> const& x);

// Load a dense (D0 x D1) row-major submatrix using an array of Simd<T, 4> (D1 <= 4, skipping
// padding if D1 != 4) from a dense (D2 x D3) row-major matrix at the the specified row and column.
template <size_t D0, size_t D1, size_t D2, size_t D3, typename T>
MOCHI_FORCE_INLINE void LoadSubmatrix(NdArray<Simd<T, 4>, D0>& out, Int2 coords, T const* x);

// Load a dense (D0 x D1) row-major submatrix using an array of Simd<T, 4> (D1 <= 4, skipping
// padding if D1 != 4) from a dense (D2 x D3) row-major matrix at the the specified row and column.
template <size_t D0, size_t D1, size_t D2, size_t D3, typename T>
MOCHI_FORCE_INLINE void LoadSubmatrix(NdArray<Simd<T, 4>, D0>& out, Int2 coords, Span<T> x);

// Store a dense (D0 x D1) row-major matrix using an array of Simd<T, 4> (D1 <= 4, skipping padding
// if D1 != 4).
template <size_t D0, size_t D1, typename T>
MOCHI_FORCE_INLINE void StoreMatrix(NdArray<T, D0, D1>& out, NdArray<Simd<T, 4>, D0> const& x);

// Store a dense (D0 x D1) row-major matrix using an array of Simd<T, 4> (D1 <= 4, skipping padding
// if D1 != 4). For this overload, the result is written to a flat pointer (memory should be
// allocated).
template <size_t D0, size_t D1, typename T>
MOCHI_FORCE_INLINE void StoreMatrix(T* outData, NdArray<Simd<T, 4>, D0> const& x);

// Store a dense (D0 x D1) row-major matrix using an array of Simd<T, 4> (D1 <= 4, skipping padding
// if D1 != 4). For this overload, the result is written to a flat Span of (D0 * D1) values.
template <size_t D0, size_t D1, typename T>
MOCHI_FORCE_INLINE void StoreMatrix(Span<T> out, NdArray<Simd<T, 4>, D0> const& x);

// Store a dense 4D row-major matrix using a 3D array of Simd<T, 4>, (D3 <= 4, skipping padding if
// D3 != 4).
template <size_t D0, size_t D1, size_t D2, size_t D3, typename T>
MOCHI_FORCE_INLINE void StoreMatrix(
    NdArray<T, D0, D1, D2, D3>& out,
    NdArray<Simd<T, 4>, D0, D1, D2> const& x);

// Store a dense (D0 x D1) row-major submatrix using an array of Simd<T, 4> (D1 <= 4, skipping
// padding if D1 != 4) in a dense (D2 x D3) row-major matrix at the the specified row and column.
template <size_t D0, size_t D1, size_t D2, size_t D3, typename T>
MOCHI_FORCE_INLINE void
StoreSubmatrix(NdArray<T, D2, D3>& out, Int2 coords, NdArray<Simd<T, 4>, D0> const& x);

// Store a dense (D0 x D1) row-major submatrix using an array of Simd<T, 4> (D1 <= 4, skipping
// padding if D1 != 4) in a dense (D2 x D3) row-major matrix at the the specified row and column.
template <size_t D0, size_t D1, size_t D2, size_t D3, typename T>
MOCHI_FORCE_INLINE void StoreSubmatrix(T* out, Int2 coords, NdArray<Simd<T, 4>, D0> const& x);

// Store a dense (D0 x D1) row-major submatrix using an array of Simd<T, 4> (D1 <= 4, skipping
// padding if D1 != 4) in a dense (D2 x D3) row-major matrix at the the specified row and column.
template <size_t D0, size_t D1, size_t D2, size_t D3, typename T>
MOCHI_FORCE_INLINE void StoreSubmatrix(Span<T> out, Int2 coords, NdArray<Simd<T, 4>, D0> const& x);

#define MOCHI_DETAILS_UNROLL_OP_SCALAR_ARRAY(result, OP, a, b) \
  if constexpr (D0 <= 4) {                                     \
    (result)[0] = a OP b[0];                                   \
    if constexpr (D0 > 1) {                                    \
      (result)[1] = a OP b[1];                                 \
    }                                                          \
    if constexpr (D0 > 2) {                                    \
      (result)[2] = a OP b[2];                                 \
    }                                                          \
    if constexpr (D0 > 3) {                                    \
      (result)[3] = a OP b[3];                                 \
    }                                                          \
  } else {                                                     \
    for (size_t i = 0; i < D0; ++i) {                          \
      (result)[i] = a OP b[i];                                 \
    }                                                          \
  }

// NdArray memberwise math operators (+=, -=, *=, /=, +, -, *, /)
#define MOCHI_DETAILS_NDARRAY_SIMD_MEMBERWISE_OP(OP_EQ, OP)                  \
  template <typename T, int N, size_t D0, size_t... DIMS>                    \
  MOCHI_FORCE_INLINE constexpr NdArray<Simd<T, N>, D0, DIMS...> operator OP( \
      NdArray<Simd<T, N>, D0, DIMS...> const& lhs, T rhs) {                  \
    return lhs OP Simd<T, N>(rhs); /* NdArray<Simd> OP Simd */               \
  }                                                                          \
  template <typename T, int N, size_t D0, size_t... DIMS>                    \
  MOCHI_FORCE_INLINE constexpr NdArray<Simd<T, N>, D0, DIMS...> operator OP( \
      T lhs, NdArray<Simd<T, N>, D0, DIMS...> const& rhs) {                  \
    NdArray<Simd<T, N>, D0, DIMS...> result{};                               \
    MOCHI_DETAILS_UNROLL_OP_SCALAR_ARRAY(                                    \
        result, OP, (Simd<T, N>{lhs}), rhs); /* Simd OP NdArray<Simd> */     \
    return result;                                                           \
  }

MOCHI_DETAILS_NDARRAY_SIMD_MEMBERWISE_OP(+=, +);
MOCHI_DETAILS_NDARRAY_SIMD_MEMBERWISE_OP(-=, -);
MOCHI_DETAILS_NDARRAY_SIMD_MEMBERWISE_OP(*=, *);
MOCHI_DETAILS_NDARRAY_SIMD_MEMBERWISE_OP(/=, /);

#undef MOCHI_DETAILS_NDARRAY_SIMD_MEMBERWISE_OP
#undef MOCHI_DETAILS_UNROLL_OP_SCALAR_ARRAY

/**************************************************************************************************
  NdArray MOCHI_FORCE_INLINEs (Simd specialization)
*/

template <typename T, size_t D0>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr bool operator==(
    NdArray<Simd<T, 4>, D0> const& lhs,
    NdArray<Simd<T, 4>, D0> const& rhs) {
  auto isEqual = VEqual(lhs[0], rhs[0]);
  if constexpr (D0 > 1) {
    isEqual &= VEqual(lhs[1], rhs[1]);
  }
  if constexpr (D0 > 2) {
    isEqual &= VEqual(lhs[2], rhs[2]);
  }
  if constexpr (D0 > 3) {
    isEqual &= VEqual(lhs[3], rhs[3]);
  }
  if constexpr (D0 > 4) {
    for (size_t i = 4; i < D0; ++i) {
      isEqual &= VEqual(lhs[i], rhs[i]);
    }
  }
  return AllTrue(isEqual);
}

template <typename T, size_t D0, size_t... DIMS>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr bool operator!=(
    NdArray<Simd<T, 4>, D0, DIMS...> const& lhs,
    NdArray<Simd<T, 4>, D0, DIMS...> const& rhs) {
  return !(lhs == rhs);
}

/**************************************************************************************************
  NdArray matrices conversions
*/

template <typename T>
MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 3> SimdSymToFull(NdArray<Simd<T, 4>, 2> const& m) {
  NdArray<Simd<T, 4>, 3> result;
  result[0] = Blend<0, 1, 1, 1>(m[0], Shuffle<3, 0, 1, 3>(m[1]));
  result[1] = Blend<1, 0, 1, 1>(m[0], Shuffle<0, 3, 2, 3>(m[1]));
  result[2] = Blend<1, 1, 0, 1>(m[0], Shuffle<1, 2, 3, 3>(m[1]));
  return result;
}

template <typename T>
MOCHI_FORCE_INLINE Simd<T, 4> SimdSymToFull(Simd<T, 4> const& m) {
  return Shuffle<0, 1, 1, 2>(m);
}

template <typename T>
MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 2> SimdFullToSym(NdArray<Simd<T, 4>, 3> const& m) {
  NdArray<Simd<T, 4>, 2> result;
  result[0] = Blend<0, 1, 1, 1>(m[0], Blend<1, 0, 1, 1>(m[1], m[2]));
  result[1] = Shuffle<1, 2, 2, 3>(m[0], m[1]);
  return result;
}

template <typename T>
MOCHI_FORCE_INLINE Simd<T, 4> SimdFullToSym(Simd<T, 4> const& m) {
  return Shuffle<0, 1, 3, 2>(m);
}

template <typename T>
MOCHI_FORCE_INLINE Real2 ToReal2(Simd<T, 4> v) {
  alignas(alignof(Simd<T, 4>)) T data[4];
  Store(data, v);
  return Real2{(real)data[0], (real)data[1]};
}

template <typename T>
MOCHI_FORCE_INLINE Real3 ToReal3(Simd<T, 4> v) {
  alignas(alignof(Simd<T, 4>)) T data[4];
  Store(data, v);
  return Real3{(real)data[0], (real)data[1], (real)data[2]};
}

template <typename T>
MOCHI_FORCE_INLINE Real4 ToReal4(Simd<T, 4> v) {
  alignas(alignof(Simd<T, 4>)) T data[4];
  Store(data, v);
  return Real4{(real)data[0], (real)data[1], (real)data[2], (real)data[3]};
}

template <typename T>
MOCHI_FORCE_INLINE NdArray<T, 2, 2> ToNdArray2x2(Simd<T, 4> m) {
  alignas(alignof(Simd<T, 4>)) T data[4];
  Store(data, m);
  return NdArray<T, 2, 2>{NdArray<T, 2>{data[0], data[1]}, NdArray<T, 2>{data[2], data[3]}};
}

template <typename T>
MOCHI_FORCE_INLINE NdArray<T, 3, 3> ToNdArray3x3(NdArray<Simd<T, 4>, 3> m) {
  NdArray<T, 3, 3> result;
  Store(result[0].data(), m[0]);
  Store(result[1].data(), m[1]);
  Store<3>(result[2].data(), m[2]);
  return result;
}

template <typename T>
MOCHI_FORCE_INLINE NdArray<T, 4, 4> ToNdArray(NdArray<Simd<T, 4>, 4> m) {
  NdArray<T, 4, 4> result;
  Store(result[0].data(), m[0]);
  Store(result[1].data(), m[1]);
  Store(result[2].data(), m[2]);
  Store(result[3].data(), m[3]);
  return result;
}

template <typename T>
MOCHI_FORCE_INLINE NdArray<T, 2, 2> ToNdArraySym2x2(Simd<T, 4> m) {
  alignas(alignof(Simd<T, 4>)) T data[4];
  Store(data, m);
  return NdArray<T, 2, 2>{NdArray<T, 2>{data[0], data[1]}, NdArray<T, 2>{data[1], data[2]}};
}

template <typename T>
MOCHI_FORCE_INLINE NdArray<T, 3, 3> ToNdArraySym3x3(NdArray<Simd<T, 4>, 2> m) {
  alignas(alignof(Simd<T, 4>)) T diag[4];
  alignas(alignof(Simd<T, 4>)) T offd[4];
  Store(diag, m[0]);
  Store(offd, m[1]);
  return NdArray<T, 3, 3>{
      NdArray<T, 3>{diag[0], offd[0], offd[1]},
      NdArray<T, 3>{offd[0], diag[1], offd[2]},
      NdArray<T, 3>{offd[1], offd[2], diag[2]}};
}

template <typename T>
MOCHI_FORCE_INLINE Simd<T, 4> ToSimd(NdArray<T, 2> const& v, T z, T w) {
  return {v[0], v[1], z, w};
}

template <typename T>
MOCHI_FORCE_INLINE Simd<T, 4> ToSimd(NdArray<T, 3> const& v, T w) {
  return {v[0], v[1], v[2], w};
}

template <typename T>
MOCHI_FORCE_INLINE Simd<T, 4> ToSimd(NdArray<T, 4> v) {
  return {v[0], v[1], v[2], v[3]};
}

template <typename T>
MOCHI_FORCE_INLINE Simd<T, 4> ToSimdMatrix(NdArray<T, 2, 2> const& m) {
  return {m[0][0], m[0][1], m[1][0], m[1][1]};
}

template <typename T>
MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 3> ToSimdMatrix(NdArray<T, 3, 3> const& m) {
  return {ToSimd(m[0]), ToSimd(m[1]), ToSimd(m[2])};
}

template <typename T>
MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 4> ToSimdMatrix(NdArray<T, 4, 4> const& m) {
  return {ToSimd(m[0]), ToSimd(m[1]), ToSimd(m[2]), ToSimd(m[3])};
}

template <typename T>
MOCHI_FORCE_INLINE Simd<T, 4> ToSimdSymMatrix(NdArray<T, 2, 2> const& m) {
  // NOTE: For the off-diagonal term, we only consider the upper-right entry (◹).
  return Simd<T, 4>{m[0][0], m[0][1], m[1][1]};
}

template <typename T>
MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 2> ToSimdSymMatrix(NdArray<T, 3, 3> const& m) {
  // NOTE: For the off-diagonal terms, we only consider the upper-right entries (◹).
  return {Simd<T, 4>(m[0][0], m[1][1], m[2][2]), Simd<T, 4>(m[0][1], m[0][2], m[1][2])};
}

template <typename T>
MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 3, 3, 3> ToSimdTensor(NdArray<T, 3, 3, 3, 3> const& t) {
  NdArray<Simd<T, 4>, 3, 3, 3> result;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      result[i][j] = ToSimdMatrix(t[i][j]);
    }
  }
  return result;
}

template <typename T>
MOCHI_FORCE_INLINE Simd<T, 4> ToSimdFromSymComponents2x2(T a00, T a01, T a11) {
  return Simd<T, 4>{a00, a01, a01, a11};
}

template <typename T>
MOCHI_FORCE_INLINE NdArray<T, 3, 3, 3, 3> ToNdArrayTensor(NdArray<Simd<T, 4>, 3, 3, 3> const& t) {
  NdArray<T, 3, 3, 3, 3> result;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      result[i][j] = ToNdArray3x3(t[i][j]);
    }
  }
  return result;
}

template <typename T>
MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 3, 3> ToSimdTensor(NdArray<T, 3, 3, 3> const& t) {
  NdArray<Simd<T, 4>, 3, 3> result;
  for (int i = 0; i < 3; ++i) {
    result[i] = ToSimdMatrix(t[i]);
  }
  return result;
}

template <typename T>
MOCHI_FORCE_INLINE NdArray<T, 3, 3, 3> ToNdArrayTensor(NdArray<Simd<T, 4>, 3, 3> const& t) {
  NdArray<T, 3, 3, 3> result;
  for (int i = 0; i < 3; ++i) {
    result[i] = ToNdArray3x3(t[i]);
  }
  return result;
}

/**************************************************************************************************
  NdArray conversions for simd <--> scalar
*/

template <size_t D0, size_t D1, typename T>
MOCHI_FORCE_INLINE void LoadMatrix(NdArray<Simd<T, 4>, D0>& out, T const* x) {
  static_assert(D1 >= 1 && D1 <= 4, "Unsupported size");
  using V = Simd<T, 4>;
  if constexpr ((D0 == 4) && (D1 == 3)) {
    // Special case for 3D tetrahedrons
    out[0] = Load<V>(x + 0 * D1);
    out[1] = Load<V>(x + 1 * D1);
    out[2] = Load<V>(x + 2 * D1);
    out[3] = Load<int(D1), V>(x + 3 * D1);
  } else {
    for (size_t i = 0; i < D0 - 1; ++i) {
      out[i] = Load<V>(x + i * D1);
    }
    out[D0 - 1] = Load<int(D1), V>(x + (D0 - 1) * D1);
  }
}

template <size_t D0, size_t D1, typename T, typename ConstOrNonConstT>
MOCHI_FORCE_INLINE void LoadMatrix(NdArray<Simd<T, 4>, D0>& out, Span<ConstOrNonConstT> x) {
  MOCHI_ASSERT_VERBOSE(x.size() == (D0 * D1), "Span size mismatch");
  LoadMatrix<D0, D1>(out, x.data());
}

template <size_t D0, size_t D1, typename T>
MOCHI_FORCE_INLINE void LoadMatrix(NdArray<Simd<T, 4>, D0>& out, NdArray<T, D0, D1> const& x) {
  LoadMatrix<D0, D1>(out, &x[0][0]);
}

template <size_t D0, size_t D1, size_t D2, size_t D3, typename T>
MOCHI_FORCE_INLINE void LoadMatrix(NdArray<Simd<T, 4>, D0, D1, D2>& out, T const* x) {
  static_assert(D3 >= 1 && D3 <= 4, "Unsupported size");
  using V = Simd<T, 4>;
  for (size_t i = 0; i < D0; ++i) {
    for (size_t j = 0; j < D1; ++j) {
      for (size_t k = 0; k < D2; ++k) {
        out[i][j][k] = Load<int(D3), V>(x + i * D1 * D2 * D3 + j * D2 * D3 + k * D3);
      }
    }
  }
}

template <size_t D0, size_t D1, size_t D2, size_t D3, typename T>
MOCHI_FORCE_INLINE void LoadMatrix(
    NdArray<Simd<T, 4>, D0, D1, D2>& out,
    NdArray<T, D0, D1, D2, D3> const& x) {
  LoadMatrix<D0, D1, D2, D3>(out, &x[0][0][0][0]);
}

template <size_t D0, size_t D1, size_t D2, size_t D3, typename T>
MOCHI_FORCE_INLINE void LoadSubmatrix(NdArray<Simd<T, 4>, D0>& out, Int2 coords, T const* x) {
  static_assert(D1 >= 1 && D1 <= 4, "Unsupported size");
  static_assert(D0 <= D2 && D1 <= D3, "Unsupported dimensions");
  MOCHI_ASSERT_VERBOSE(coords[0] + D0 <= D2, "Size mismatch");
  MOCHI_ASSERT_VERBOSE(coords[1] + D1 <= D3, "Size mismatch");
  MOCHI_ASSERT_VERBOSE(coords[0] >= 0 && coords[1] >= 0, "Invalid coordinates");
  using V = Simd<T, 4>;
  for (size_t i = 0; i < D0 - 1; ++i) {
    out[i] = Load<V>(x + (coords[0] + i) * D3 + coords[1]);
  }
  out[D0 - 1] = Load<int(D1), V>(x + (coords[0] + D0 - 1) * D3 + coords[1]);
}

template <size_t D0, size_t D1, size_t D2, size_t D3, typename T, typename ConstOrNonConstT>
MOCHI_FORCE_INLINE void
LoadSubmatrix(NdArray<Simd<T, 4>, D0>& out, Int2 coords, Span<ConstOrNonConstT> x) {
  MOCHI_ASSERT_VERBOSE(x.size() == (D2 * D3), "Span size mismatch");
  LoadSubmatrix<D0, D1, D2, D3>(out, coords, x.data());
}

template <size_t D0, size_t D1, size_t D2, size_t D3, typename T>
MOCHI_FORCE_INLINE void
LoadSubmatrix(NdArray<Simd<T, 4>, D0>& out, Int2 coords, NdArray<T, D2, D3> const& x) {
  LoadSubmatrix<D0, D1, D2, D3>(out, coords, &x[0][0]);
}

template <size_t D0, size_t D1, typename T>
MOCHI_FORCE_INLINE void StoreMatrix(T* out, NdArray<Simd<T, 4>, D0> const& x) {
  static_assert(D1 >= 1 && D1 <= 4, "Unsupported size");
  if constexpr ((D0 == 4) && (D1 == 3)) {
    // Special case for 3D tetrahedrons
    Store(out + 0 * D1, x[0]);
    Store(out + 1 * D1, x[1]);
    Store(out + 2 * D1, x[2]);
    Store<int(D1)>(out + 3 * D1, x[3]);
  } else {
    for (size_t i = 0; i < D0 - 1; ++i) {
      Store(out + i * D1, x[i]);
    }
    Store<int(D1)>(out + (D0 - 1) * D1, x[D0 - 1]);
  }
}

template <size_t D0, size_t D1, typename T>
MOCHI_FORCE_INLINE void StoreMatrix(Span<T> out, NdArray<Simd<T, 4>, D0> const& x) {
  MOCHI_ASSERT_VERBOSE(out.size() == (D0 * D1), "Span size mismatch");
  StoreMatrix<D0, D1>(out.data(), x);
}

template <size_t D0, size_t D1, typename T>
MOCHI_FORCE_INLINE void StoreMatrix(NdArray<T, D0, D1>& out, NdArray<Simd<T, 4>, D0> const& x) {
  StoreMatrix<D0, D1>(&out[0][0], x);
}

template <size_t D0, size_t D1, size_t D2, size_t D3, typename T>
MOCHI_FORCE_INLINE void StoreMatrix(T* out, NdArray<Simd<T, 4>, D0, D1, D2> const& x) {
  static_assert(D3 >= 1 && D3 <= 4, "Unsupported size");
  for (size_t i = 0; i < D0; ++i) {
    for (size_t j = 0; j < D1; ++j) {
      for (size_t k = 0; k < D2; ++k) {
        Store<int(D3)>(out + i * D1 * D2 * D3 + j * D2 * D3 + k * D3, x[i][j][k]);
      }
    }
  }
}

template <size_t D0, size_t D1, size_t D2, size_t D3, typename T>
MOCHI_FORCE_INLINE void StoreMatrix(
    NdArray<T, D0, D1, D2, D3>& out,
    NdArray<Simd<T, 4>, D0, D1, D2> const& x) {
  StoreMatrix<D0, D1, D2, D3>(&out[0][0][0][0], x);
}

template <size_t D0, size_t D1, size_t D2, size_t D3, typename T>
MOCHI_FORCE_INLINE void StoreSubmatrix(T* out, Int2 coords, NdArray<Simd<T, 4>, D0> const& x) {
  static_assert(D1 >= 1 && D1 <= 4, "Unsupported size");
  static_assert(D0 <= D2 && D1 <= D3, "Unsupported dimensions");
  MOCHI_ASSERT_VERBOSE(coords[0] + D0 <= D2, "Size mismatch");
  MOCHI_ASSERT_VERBOSE(coords[1] + D1 <= D3, "Size mismatch");
  MOCHI_ASSERT_VERBOSE(coords[0] >= 0 && coords[1] >= 0, "Invalid coordinates");
  for (size_t i = 0; i < D0; ++i) {
    Store<int(D1)>(out + (coords[0] + i) * D3 + coords[1], x[i]);
  }
}

template <size_t D0, size_t D1, size_t D2, size_t D3, typename T>
MOCHI_FORCE_INLINE void StoreSubmatrix(Span<T> out, Int2 coords, NdArray<Simd<T, 4>, D0> const& x) {
  MOCHI_ASSERT_VERBOSE(out.size() == (D2 * D3), "Span size mismatch");
  StoreSubmatrix<D0, D1, D2, D3>(out.data(), coords, x);
}

template <size_t D0, size_t D1, size_t D2, size_t D3, typename T>
MOCHI_FORCE_INLINE void
StoreSubmatrix(NdArray<T, D2, D3>& out, Int2 coords, NdArray<Simd<T, 4>, D0> const& x) {
  StoreSubmatrix<D0, D1, D2, D3>(&out[0][0], coords, x);
}

/**************************************************************************************************
  BroadcastEach
*/

namespace ndarray_details {
template <class V, size_t D0, size_t... DIMS, size_t... I>
[[nodiscard]] MOCHI_FORCE_INLINE auto BroadcastEach( // TODO: Make this a lambda with C++20
    NdArray<typename V::Scalar, D0, DIMS...> const& a,
    std::index_sequence<I...>) {
  if constexpr (sizeof...(DIMS) == 0) { // Last dimension is a Simd broadcast
    return NdArray<V, D0, DIMS...>{Broadcast<V>(a[I])...};
  } else {
    return NdArray<V, D0, DIMS...>{BroadcastEach<V>(a[I])...};
  }
}
template <class V, int N, size_t... I>
[[nodiscard]] MOCHI_FORCE_INLINE auto BroadcastEach( // TODO: Make this a lambda with C++20
    Simd<typename V::Scalar, N> const& a, std::index_sequence<I...>) {
  if constexpr (V::kSize == N) {
    return NdArray<V, (size_t)N>{Broadcast<(int)I>(a)...}; // Broadcast I'th to same Simd size
  } else {
    return NdArray<V, (size_t)N>{Broadcast<V>(Get<I>(a))...}; // Broadcast I'th to other Simd size
  }
}
} // namespace ndarray_details

// Broadcast each member of an NdArray to a Simd vector.
// Example:
//    auto pt = Real3{x, y, z};
//    auto v = Broadcast<Vec4r>(pt); // v = {{x, x, x, x}, {y, y, y, y}, {z, z, z, z}}
//
template <class V, size_t D0, size_t... DIMS, MOCHI_CONCEPT(IsSimd<V>)>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<V, D0, DIMS...> BroadcastEach(
    NdArray<typename V::Scalar, D0, DIMS...> const& a) {
  return ndarray_details::BroadcastEach<V>(a, std::make_index_sequence<D0>());
}

// Broadcast each member a Simd vector to its own Simd vector
// Example:
//    auto pt = Vec4r{a, b, c, d};
//    auto v = Broadcast<Vec4r>(pt); // v = {{a, a, a, a}, {b, b, b, b}, {c, c, c, c}, {d, d, d, d}}
//
template <class V, int N, MOCHI_CONCEPT(IsSimd<V>)>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<V, N> BroadcastEach(Simd<typename V::Scalar, N> const& a) {
  return ndarray_details::BroadcastEach<V>(a, std::make_index_sequence<(size_t)N>());
}

// Broadcast just the first 3 members of a Simd vector to its own Simd vector
// Example:
//    auto pt = Vec4r{a, b, c, d};
//    auto v = Broadcast3<Vec4r>(pt); // v = {{a, a, a, a}, {b, b, b, b}, {c, c, c, c}}
//
template <class V, int N, MOCHI_CONCEPT(IsSimd<V>&& N >= 3)>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<V, 3> Broadcast3(Simd<typename V::Scalar, N> const& a) {
  if constexpr (V::kSize == N) {
    return NdArray<V, 3>{Broadcast<0>(a), Broadcast<1>(a), Broadcast<2>(a)};
  } else {
    return NdArray<V, 3>{Broadcast<V>(Get<0>(a)), Broadcast<V>(Get<1>(a)), Broadcast<V>(Get<2>(a))};
  }
}

// Broadcast just the first 3x3 portion of a Simd matrix, each member to its own vector.
// Example:
//    auto mat = VMatrix4x4{row0, row1, row2, row3};
//    auto vmat = Broadcast3x3<Vec4r>(mat);
//    // vmat = {Broadcast3<Vec4r>(mat[0]), Broadcast3<Vec4r>(mat[1]), Broadcast3<Vec4r>(mat[2])};
//
template <class V, size_t D0, int D1, MOCHI_CONCEPT(IsSimd<V>&& D0 >= 3 && D1 >= 3)>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<V, 3, 3> Broadcast3x3(
    NdArray<Simd<typename V::Scalar, D1>, D0> const& a) {
  return NdArray<V, 3, 3>{Broadcast3<V>(a[0]), Broadcast3<V>(a[1]), Broadcast3<V>(a[2])};
}

/**************************************************************************************************
  LoadTransposed / StoreTransposed
*/

// Load (kTupleCount * 3) values in transposed order (kTupleCount == V::kSize by default).
// In other words:
//    {{x0, y0, z0}, {x1, y1, z1}, ...} --> {{x0, x1, ...}, {y0, y1, ...}, {z0, z1, ...}}
template <int kTupleCount = -1, class V, MOCHI_CONCEPT(IsSimd<V>)>
MOCHI_FORCE_INLINE void LoadTransposed(typename V::Scalar const* ptr, NdArray<V, 3>& out) {
  auto constexpr kTupleCount_ = (kTupleCount == -1) ? V::kSize : kTupleCount;
  LoadTransposed<kTupleCount_>(ptr, out[0], out[1], out[2]);
}

// Store (kTupleCount * 3) values in transposed order (kTupleCount == V::kSize by default).
// In other words:
//    {{x0, x1, ...}, {y0, y1, ...}, {z0, z1, ...}} --> {{x0, y0, z0}, {x1, y1, z1}, ...}
template <int kTupleCount = -1, class V, MOCHI_CONCEPT(IsSimd<V>)>
MOCHI_FORCE_INLINE void StoreTransposed(typename V::Scalar* ptr, NdArray<V, 3> const& src) {
  auto constexpr kTupleCount_ = (kTupleCount == -1) ? V::kSize : kTupleCount;
  StoreTransposed<kTupleCount_>(
      ptr, src[0], src[1], src[2]); // TODO: Support arbitrary size when C++20 makes that easier
}

} // namespace mochi
