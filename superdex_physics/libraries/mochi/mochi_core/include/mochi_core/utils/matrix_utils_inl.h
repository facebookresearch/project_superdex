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

#include <mochi_core/utils/matrix_utils.h> // for intellisense

#include <cmath>

namespace mochi {

/**************************************************************************************************
  Frobenius Norm
*/

template <typename T, size_t N, size_t M>
MOCHI_FORCE_INLINE constexpr T NormSqr(NdArray<T, N, M> const& a) {
  T accum = T(0);
  for (int i = 0; i < N; ++i) {
    accum += Dot(a[i], a[i]);
  }
  return accum;
}

template <typename T, size_t N, size_t M>
MOCHI_FORCE_INLINE T Norm(NdArray<T, N, M> const& a) {
  return Sqrt(NormSqr(a));
}

template <typename T, size_t D0, int D1, MOCHI_CONCEPT_DEF(D0 >= 3 && D1 >= 3)>
MOCHI_FORCE_INLINE T NormSqr3x3(NdArray<Simd<T, D1>, D0> const& a) {
  return HSum<3>(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}

template <typename T, size_t D0, int D1, MOCHI_CONCEPT_DEF(D0 >= 3 && D1 >= 3)>
MOCHI_FORCE_INLINE T Norm3x3(NdArray<Simd<T, D1>, D0> const& a) {
  return Sqrt(NormSqr3x3<T, D0, D1>(a));
}

/**************************************************************************************************
  Matrix-Vector Dot Product
*/

template <typename T, size_t N, size_t M>
MOCHI_FORCE_INLINE constexpr NdArray<T, N> DotMatVec(
    NdArray<T, N, M> const& a,
    NdArray<T, M> const& b) {
  NdArray<T, N> result = {};
  for (size_t i = 0; i < result.size(); ++i) {
    result[i] = Dot(a[i], b);
  }
  return result;
}

template <typename T, size_t N, size_t M>
MOCHI_FORCE_INLINE void DotMatVec(NdArray<T, N, M> const& a, Span<T const> b, Span<T> out) {
  MOCHI_ASSERT_VERBOSE((M == b.size()) && (out.size() == N), "Size mismatch");
  for (size_t i = 0; i < out.size(); ++i) {
    out[i] = Dot(b, a[i]);
  }
}

template <typename T, size_t N, size_t M>
MOCHI_FORCE_INLINE constexpr NdArray<T, N> DotMatVec(NdArray<T, N, M> const& a, Span<T const> b) {
  NdArray<T, N> result = {};
  DotMatVec(a, b, Span<T>(result));
  return result;
}

template <typename T>
MOCHI_FORCE_INLINE constexpr NdArray<T, 3> DotMatVec(
    NdArray<T, 3, 3> const& a,
    NdArray<T, 3> const& b) {
  return NdArray<T, 3>{
      a[0][0] * b[0] + a[0][1] * b[1] + a[0][2] * b[2],
      a[1][0] * b[0] + a[1][1] * b[1] + a[1][2] * b[2],
      a[2][0] * b[0] + a[2][1] * b[1] + a[2][2] * b[2]};
}

template <size_t N, typename T, size_t D0>
MOCHI_FORCE_INLINE Simd<T, 4> DotMatVec3xN(NdArray<Simd<T, 4>, D0> const& m, Simd<T, 4> v) {
  static_assert(D0 >= 3, "Requires a matrix with at least 3 rows");
  auto const a = VDot<N>(m[0], v);
  auto const b = VDot<N>(m[1], v);
  auto const c = VDot<N>(m[2], v);
  return Shuffle<0, 1, 0, 1>(Blend<0, 1, 0, 0>(a, b), c);
}

template <typename T, size_t D0>
MOCHI_FORCE_INLINE Simd<T, 4> DotMatVec3x3(NdArray<Simd<T, 4>, D0> const& m, Simd<T, 4> v) {
  return DotMatVec3xN<3>(m, v);
}

template <typename T>
MOCHI_FORCE_INLINE Simd<T, 4> DotMatVec4x4(NdArray<Simd<T, 4>, 4> const& m, Simd<T, 4> v) {
  auto a = VDot(m[0], v);
  auto b = VDot(m[1], v);
  auto c = VDot(m[2], v);
  auto d = VDot(m[3], v);
  auto ab = Shuffle<0, 0, 0, 0>(a, b); // ac = (a[0], a[0], b[0], b[0])
  auto cd = Shuffle<0, 0, 0, 0>(c, d); // bd = (c[0], c[0], d[0], d[0])
  return Shuffle<0, 2, 0, 2>(ab, cd); // return (a[0], b[0], c[0], d[0])
}

/**************************************************************************************************
  Vector-Matrix Dot Product
*/

template <typename T, size_t N, size_t M>
MOCHI_FORCE_INLINE constexpr NdArray<T, M> DotVecMat(
    NdArray<T, N> const& a,
    NdArray<T, N, M> const& b) {
  if constexpr (N == 3 && M == 3) {
    return NdArray<T, 3>{
        a[0] * b[0][0] + a[1] * b[1][0] + a[2] * b[2][0],
        a[0] * b[0][1] + a[1] * b[1][1] + a[2] * b[2][1],
        a[0] * b[0][2] + a[1] * b[1][2] + a[2] * b[2][2]};
  } else {
    return DotMatVec(Transpose(b), a);
  }
}

template <typename T>
MOCHI_FORCE_INLINE Simd<T, 4> DotVecMat2x3(Simd<T, 4> v, NdArray<Simd<T, 4>, 2> const& m) {
  // |u v| | a b c | = | ua + vd |
  //       | d e f |   | ub + ve |
  //                   | uc + vf |
  Simd<T, 4> temp0 = Broadcast<0>(v) * m[0]; // (ua, ub, uc, ?)
  Simd<T, 4> temp1 = Broadcast<1>(v) * m[1]; // (vd, ve, vf, ?)
  return temp0 + temp1; // (ua + vd, ub + ve, uc + vf, ?)
}

template <typename T, size_t D0>
MOCHI_FORCE_INLINE Simd<T, 4> DotVecMat3x3(Simd<T, 4> a, NdArray<Simd<T, 4>, D0> const& b) {
  static_assert(D0 == 3 || D0 == 4, "Requires a matrix with 3 or 4 rows");
  return Broadcast<0>(a) * b[0] + Broadcast<1>(a) * b[1] + Broadcast<2>(a) * b[2];
}

template <typename T>
MOCHI_FORCE_INLINE Simd<T, 4> DotVecMat4x4(Simd<T, 4> a, NdArray<Simd<T, 4>, 4> const& b) {
  return Broadcast<0>(a) * b[0] + Broadcast<1>(a) * b[1] + Broadcast<2>(a) * b[2] +
      Broadcast<3>(a) * b[3];
}

/**************************************************************************************************
  Matrix-Matrix Dot Product
*/

template <typename T, size_t N, size_t M, size_t L>
MOCHI_FORCE_INLINE constexpr NdArray<T, N, L> Dot(
    NdArray<T, N, M> const& a,
    NdArray<T, M, L> const& b) {
  NdArray<T, N, L> result = {};
  NdArray<T, L, M> bT = Transpose(b);
  for (int n = 0; n < N; ++n) {
    result[n] = DotMatVec(bT, a[n]);
  }
  return result;
}

template <typename T>
MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 4> Dot4x4(
    NdArray<Simd<T, 4>, 4> const& a,
    NdArray<Simd<T, 4>, 4> const& b) {
  return NdArray<Simd<T, 4>, 4>{
      DotVecMat4x4(a[0], b), DotVecMat4x4(a[1], b), DotVecMat4x4(a[2], b), DotVecMat4x4(a[3], b)};
}

template <typename T>
MOCHI_FORCE_INLINE constexpr NdArray<T, 3, 3> Dot(
    NdArray<T, 3, 3> const& a,
    NdArray<T, 3, 3> const& b) {
  return {
      NdArray<T, 3>{
          a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0],
          a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1],
          a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2]},
      NdArray<T, 3>{
          a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0],
          a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1],
          a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2]},
      NdArray<T, 3>{
          a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0],
          a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1],
          a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2]}};
}

template <typename T, size_t D0A, size_t D0B>
MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 3> Dot3x3(
    NdArray<Simd<T, 4>, D0A> const& a,
    NdArray<Simd<T, 4>, D0B> const& b) {
  return {DotVecMat3x3(a[0], b), DotVecMat3x3(a[1], b), DotVecMat3x3(a[2], b)};
}

template <typename T>
MOCHI_FORCE_INLINE Simd<T, 4> Dot2x2(Simd<T, 4> const& a, Simd<T, 4> const& b) {
  Simd<T, 4> const a0 = Shuffle<0, 1, 3, 2>(a);
  Simd<T, 4> const b0 = Shuffle<0, 3, 2, 1>(b);
  Simd<T, 4> const a1 = Shuffle<1, 0, 2, 3>(a);
  Simd<T, 4> const b1 = Shuffle<2, 1, 0, 3>(b);
  return a0 * b0 + a1 * b1;
}

/**************************************************************************************************
  Outer Product
*/

template <typename T>
MOCHI_FORCE_INLINE constexpr NdArray<T, 2, 2> Outer(
    NdArray<T, 2> const& a,
    NdArray<T, 2> const& b) {
  return {NdArray<T, 2>{a[0] * b[0], a[0] * b[1]}, NdArray<T, 2>{a[1] * b[0], a[1] * b[1]}};
}

template <typename T>
MOCHI_FORCE_INLINE constexpr NdArray<T, 3, 2> Outer(
    NdArray<T, 3> const& a,
    NdArray<T, 2> const& b) {
  return NdArray<T, 3, 2>{
      NdArray<T, 2>{a[0] * b[0], a[0] * b[1]},
      NdArray<T, 2>{a[1] * b[0], a[1] * b[1]},
      NdArray<T, 2>{a[2] * b[0], a[2] * b[1]}};
}

template <typename T>
MOCHI_FORCE_INLINE constexpr NdArray<T, 3, 3> Outer(
    NdArray<T, 3> const& a,
    NdArray<T, 3> const& b) {
  return {
      NdArray<T, 3>{a[0] * b[0], a[0] * b[1], a[0] * b[2]},
      NdArray<T, 3>{a[1] * b[0], a[1] * b[1], a[1] * b[2]},
      NdArray<T, 3>{a[2] * b[0], a[2] * b[1], a[2] * b[2]}};
}

// Outer product of 3 component vectors (assumed 4th component unused)
template <typename T>
MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 3> Outer3(Simd<T, 4> a, Simd<T, 4> b) {
  return NdArray<Simd<T, 4>, 3>{
      Shuffle<0, 0, 0, 0>(a) * b, // {a[0] * b[0], a[0] * b[1], a[0] * b[2], a[0] * b[3] }
      Shuffle<1, 1, 1, 1>(a) * b, // {a[1] * b[0], a[1] * b[1], a[1] * b[2], a[1] * b[3] }
      Shuffle<2, 2, 2, 2>(a) * b, // {a[2] * b[0], a[2] * b[1], a[2] * b[2], a[2] * b[3] }
  };
}

template <typename T>
MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 3, 3> Outer3(
    Simd<T, 4> const& vec,
    NdArray<Simd<T, 4>, 3> const& mat) {
  NdArray<Simd<T, 4>, 3, 3> outProd;
  outProd[0] = mat * Broadcast<0>(vec);
  outProd[1] = mat * Broadcast<1>(vec);
  outProd[2] = mat * Broadcast<2>(vec);
  return outProd;
}

// Outer product of a 3x3 matrix and a 3x1 vector returning a 3x3x3 3rd-order tensor
template <typename T>
MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 3, 3> Outer3(
    NdArray<Simd<T, 4>, 3> const& mat,
    Simd<T, 4> const& vec) {
  NdArray<Simd<T, 4>, 3, 3> result;
  for (int i = 0; i < 3; ++i) {
    result[i] = Outer3(mat[i], vec);
  }
  return result;
}

template <typename T>
MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 3, 3, 3> Outer3(
    NdArray<Simd<T, 4>, 3, 3> const& ten,
    Simd<T, 4> const& vec) {
  NdArray<Simd<T, 4>, 3, 3, 3> outProd;
  for (int i = 0; i < 3; ++i) {
    outProd[i][0] = ten[i] * Broadcast<0>(vec);
    outProd[i][1] = ten[i] * Broadcast<1>(vec);
    outProd[i][2] = ten[i] * Broadcast<2>(vec);
  }
  return outProd;
}

template <typename T>
MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 3, 3, 3> Outer3(
    Simd<T, 4> const& vec,
    NdArray<Simd<T, 4>, 3, 3> const& ten) {
  NdArray<Simd<T, 4>, 3, 3, 3> outProd;
  for (int i = 0; i < 3; ++i) {
    outProd[0][i] = ten[i] * Broadcast<0>(vec);
    outProd[1][i] = ten[i] * Broadcast<1>(vec);
    outProd[2][i] = ten[i] * Broadcast<2>(vec);
  }
  return outProd;
}

template <typename T>
MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 3, 3, 3> Outer3(
    NdArray<Simd<T, 4>, 3> const& mat0,
    NdArray<Simd<T, 4>, 3> const& mat1) {
  NdArray<Simd<T, 4>, 3, 3, 3> outProd;
  for (int i = 0; i < 3; ++i) {
    outProd[i][0] = mat0 * Broadcast<0>(mat1[i]);
    outProd[i][1] = mat0 * Broadcast<1>(mat1[i]);
    outProd[i][2] = mat0 * Broadcast<2>(mat1[i]);
  }
  return outProd;
}

template <typename T>
MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 2> OuterSym3(Simd<T, 4> a, Simd<T, 4> b) {
  // {Vec4r{a0b0, a1b1, a2b2, *}, Vec4r{a0b1, a0b2, a1b2, *}}
  return {a * b, Shuffle<0, 0, 1, 3>(a) * Shuffle<1, 2, 2, 3>(b)};
}

template <typename T>
MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 2, 2> Outer2(Simd<T, 4> const& a, Simd<T, 4> const& b) {
  NdArray<Simd<T, 4>, 2, 2> result;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      result[i][j] = a[2 * i + j] * b;
    }
  }
  return result;
}

/**************************************************************************************************
  Inner Product
*/

template <typename T>
MOCHI_FORCE_INLINE T Colon3x3(NdArray<Simd<T, 4>, 3> const& A, NdArray<Simd<T, 4>, 3> const& B) {
  T result = {};
  for (int i = 0; i < 3; ++i) {
    result += Dot<3>(A[i], B[i]);
  }
  return result;
}

template <typename T, size_t N, size_t M>
MOCHI_FORCE_INLINE constexpr T Colon(NdArray<T, N, M> const& A, NdArray<T, N, M> const& B) {
  T result = T{};
  for (size_t i = 0; i < N; ++i) {
    result += Dot(A[i], B[i]);
  }
  return result;
}

/**************************************************************************************************
  Invert
*/

template <typename T>
MOCHI_FORCE_INLINE constexpr NdArray<T, 2, 2> Invert(NdArray<T, 2, 2> const& A) {
  return Invert(A, Det(A));
}

template <typename T>
MOCHI_FORCE_INLINE constexpr NdArray<T, 2, 2> Invert(NdArray<T, 2, 2> const& A, T det) {
  MOCHI_ASSERT_VERBOSE(!NearEqual(det, T(0), T(ScalarType<T>(1e-16))), "Non-invertible matrix.");
  T const invDet = T(1) / det;
  return {
      NdArray<T, 2>{A[1][1] * invDet, -A[0][1] * invDet},
      NdArray<T, 2>{-A[1][0] * invDet, A[0][0] * invDet}};
}

template <typename T>
MOCHI_FORCE_INLINE constexpr NdArray<T, 3, 3> Invert(NdArray<T, 3, 3> const& A) {
  return Invert(A, Det(A));
}

template <typename T>
MOCHI_FORCE_INLINE constexpr NdArray<T, 3, 3> Invert(NdArray<T, 3, 3> const& A, T det) {
  MOCHI_ASSERT_VERBOSE(!NearEqual(det, T(0), T(ScalarType<T>(1e-16))), "Non-invertible matrix.");
  NdArray<T, 3, 3> cofactorTranspose{
      NdArray<T, 3>{
          A[1][1] * A[2][2] - A[1][2] * A[2][1],
          A[0][2] * A[2][1] - A[0][1] * A[2][2],
          A[0][1] * A[1][2] - A[0][2] * A[1][1]},
      NdArray<T, 3>{
          A[1][2] * A[2][0] - A[1][0] * A[2][2],
          A[0][0] * A[2][2] - A[0][2] * A[2][0],
          A[0][2] * A[1][0] - A[0][0] * A[1][2]},
      NdArray<T, 3>{
          A[1][0] * A[2][1] - A[1][1] * A[2][0],
          A[0][1] * A[2][0] - A[0][0] * A[2][1],
          A[0][0] * A[1][1] - A[0][1] * A[1][0]}};
  return cofactorTranspose / det;
}

template <typename T>
inline void PseudoInvert(NdArray<T, 3, 2> const& a, NdArray<T, 2, 3>* outInv, T* outDet) {
  NdArray<T, 2, 3> const at = Transpose(a);
  NdArray<T, 3, 3> const coef_1 = Outer(at[0], at[1]) - Outer(at[1], at[0]);
  T const coef_2 = Dot(DotVecMat(at[0], coef_1), at[1]);
  (*outInv)[0] = DotMatVec(coef_1, at[1]) / coef_2;
  (*outInv)[1] = -DotMatVec(coef_1, at[0]) / coef_2;
  *outDet = Norm(Cross(at[0], at[1]));
}

template <typename T>
MOCHI_FORCE_INLINE constexpr T
Minor4x4(T const m[16], int r0, int r1, int r2, int c0, int c1, int c2) {
  return m[4 * r0 + c0] * (m[4 * r1 + c1] * m[4 * r2 + c2] - m[4 * r2 + c1] * m[4 * r1 + c2]) -
      m[4 * r0 + c1] * (m[4 * r1 + c0] * m[4 * r2 + c2] - m[4 * r2 + c0] * m[4 * r1 + c2]) +
      m[4 * r0 + c2] * (m[4 * r1 + c0] * m[4 * r2 + c1] - m[4 * r2 + c0] * m[4 * r1 + c1]);
}

template <typename T>
inline constexpr void Cofactor4x4(T const m[16], T adjOut[16]) {
  adjOut[0] = Minor4x4(m, 1, 2, 3, 1, 2, 3);
  adjOut[1] = -Minor4x4(m, 0, 2, 3, 1, 2, 3);
  adjOut[2] = Minor4x4(m, 0, 1, 3, 1, 2, 3);
  adjOut[3] = -Minor4x4(m, 0, 1, 2, 1, 2, 3);
  adjOut[4] = -Minor4x4(m, 1, 2, 3, 0, 2, 3);
  adjOut[5] = Minor4x4(m, 0, 2, 3, 0, 2, 3);
  adjOut[6] = -Minor4x4(m, 0, 1, 3, 0, 2, 3);
  adjOut[7] = Minor4x4(m, 0, 1, 2, 0, 2, 3);
  adjOut[8] = Minor4x4(m, 1, 2, 3, 0, 1, 3);
  adjOut[9] = -Minor4x4(m, 0, 2, 3, 0, 1, 3);
  adjOut[10] = Minor4x4(m, 0, 1, 3, 0, 1, 3);
  adjOut[11] = -Minor4x4(m, 0, 1, 2, 0, 1, 3);
  adjOut[12] = -Minor4x4(m, 1, 2, 3, 0, 1, 2);
  adjOut[13] = Minor4x4(m, 0, 2, 3, 0, 1, 2);
  adjOut[14] = -Minor4x4(m, 0, 1, 3, 0, 1, 2);
  adjOut[15] = Minor4x4(m, 0, 1, 2, 0, 1, 2);
}

template <typename T>
inline constexpr T Det4x4(T const m[16]) {
  return m[0] * Minor4x4(m, 1, 2, 3, 1, 2, 3) - m[1] * Minor4x4(m, 1, 2, 3, 0, 2, 3) +
      m[2] * Minor4x4(m, 1, 2, 3, 0, 1, 3) - m[3] * Minor4x4(m, 1, 2, 3, 0, 1, 2);
}

template <typename T>
inline constexpr void InvertRowMajor4x4(T const m[16], T invOut[16]) {
  Cofactor4x4(m, invOut);

  T det = Det4x4(m);

  MOCHI_ASSERT_VERBOSE(!NearEqual(det, T(0), T(ScalarType<T>(1e-16))), "Non-invertible matrix.");

  T inv_det = T(1) / det;
  for (int i = 0; i < 16; ++i) {
    invOut[i] = invOut[i] * inv_det;
  }
}

template <typename T>
MOCHI_FORCE_INLINE constexpr NdArray<T, 4, 4> Invert(NdArray<T, 4, 4> const& A) {
  NdArray<T, 4, 4> out;
  InvertRowMajor4x4(&A[0][0], &out[0][0]);
  return out;
}

template <typename T>
MOCHI_FORCE_INLINE constexpr NdArray<Simd<T, 4>, 4> Invert4x4(NdArray<Simd<T, 4>, 4> const& A) {
  // This calls the non-SIMD implementation above. Could probably be much faster with a proper SIMD
  // implementation.
  NdArray<Simd<T, 4>, 4> out;
  static_assert(sizeof(NdArray<Simd<T, 4>, 4>) == sizeof(NdArray<T, 4, 4>));
  InvertRowMajor4x4(reinterpret_cast<T const*>(&A[0]), reinterpret_cast<T*>(&out[0]));
  return out;
}

template <typename T>
MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 3> Invert3x3(NdArray<Simd<T, 4>, 3> const& mat) {
  return Invert3x3(mat, VDet3x3(mat));
}

template <typename T>
MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 3> Invert3x3(
    NdArray<Simd<T, 4>, 3> const& mat,
    Simd<T, 4> det) {
  MOCHI_ASSERT_VERBOSE(!NearEqual(Get0(det), T(0), T(1.e-16)), "Non-invertible matrix");
  return Transpose3x3(Cofactor3x3(mat)) / det;
}

template <typename T>
MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 3> Invert3x3(NdArray<Simd<T, 4>, 3> const& mat, T det) {
  return Invert3x3(mat, Simd<T, 4>{det});
}

template <typename T>
MOCHI_FORCE_INLINE Simd<T, 4> Invert2x2(Simd<T, 4> const& mat, T det) {
  return (T{1} / det) * Neg<false, true, true, false>(Shuffle<3, 1, 2, 0>(mat));
}

// Implementation inspired by:
// https://lxjk.github.io/2017/09/03/Fast-4x4-Matrix-Inverse-with-SSE-SIMD-Explained.html
//
template <typename T>
MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 4> InvertTransformationTransposed(
    NdArray<Simd<T, 4>, 4> const& mat) {
#if MOCHI_ASSERT_VERBOSE_ENABLED
  constexpr T tol = T{10} * kDefaultNearEqualEpsilon<T>;

  MOCHI_ASSERT_VERBOSE(
      (mat[0][3] == 0) && (mat[1][3] == 0) && (mat[2][3] == 0) && (mat[3][3] == 1),
      "Expected a transformation matrix where the last column is {0, 0, 0, 1}");

  auto n0 = Norm<3>(mat[0]);
  auto n1 = Norm<3>(mat[1]);
  auto n2 = Norm<3>(mat[2]);
  MOCHI_ASSERT_VERBOSE(
      NearZero(Dot<3>(mat[0], mat[1]), tol * n0 * n1) &&
          NearZero(Dot<3>(mat[1], mat[2]), tol * n1 * n2) &&
          NearZero(Dot<3>(mat[2], mat[0]), tol * n0 * n2),
      "Expected a transformation matrix with orthogonal basis vectors");

#endif // MOCHI_ASSERT_VERBOSE_ENABLED

  NdArray<Simd<T, 4>, 4> r;

  // Transpose 3x3. We know m03 = m13 = m23 = 0
  auto t0 = Shuffle<0, 1, 0, 1>(mat[0], mat[1]); // 00, 01, 10, 11
  auto t1 = Shuffle<2, 3, 2, 3>(mat[0], mat[1]); // 02, 03, 12, 13
  r[0] = Shuffle<0, 2, 0, 3>(t0, mat[2]); // 00, 10, 20, 23(=0)
  r[1] = Shuffle<1, 3, 1, 3>(t0, mat[2]); // 01, 11, 21, 23(=0)
  r[2] = Shuffle<0, 2, 2, 3>(t1, mat[2]); // 02, 12, 22, 23(=0)

  // Invert scale
  auto sizeSqr = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
  MOCHI_ASSERT_VERBOSE(
      AllTrue<3>(sizeSqr > Sqr(tol)),
      "Transformation matrix has degenerate scale on at least one axis");
  auto rSizeSqr = T(1) / ToSimdPoint(sizeSqr);
  r[0] *= rSizeSqr;
  r[1] *= rSizeSqr;
  r[2] *= rSizeSqr;

  // Last row (translation)
  r[3] = r[0] * Broadcast<0>(mat[3]) + r[1] * Broadcast<1>(mat[3]) + r[2] * Broadcast<2>(mat[3]);
  r[3] = ToSimdPoint(-r[3]);

  return r;
}

template <typename T>
MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 4> InvertTransformation(NdArray<Simd<T, 4>, 4> const& mat) {
  // This performs 8 SIMD shuffles to transpose the input and another 8 to transpose the result
  // (thus about 16 cycles slower than InvertTransformationTransposed). This could be improved if we
  // think it is worth the effort.
  return Transpose4x4(InvertTransformationTransposed(Transpose4x4(mat)));
}

/**************************************************************************************************
  Matrix Builders
*/

template <size_t N, typename T>
MOCHI_FORCE_INLINE constexpr NdArray<T, N, N> DiagonalMatrix(T valueOnDiagonal) {
  NdArray<T, N, N> result = {};
  for (size_t i = 0; i < result.size(); ++i) {
    result[i][i] = valueOnDiagonal;
  }
  return result;
}

template <size_t N, typename T>
MOCHI_FORCE_INLINE constexpr NdArray<T, N, N> DiagonalMatrix(NdArray<T, N> const& diagonalVector) {
  NdArray<T, N, N> result = {};
  for (size_t i = 0; i < result.size(); ++i) {
    result[i][i] = diagonalVector[i];
  }
  return result;
}

template <typename T>
MOCHI_FORCE_INLINE constexpr NdArray<T, 2, 2> SymMatrix2x2(T a00, T a01, T a11) {
  return {NdArray<T, 2>{a00, a01}, NdArray<T, 2>{a01, a11}};
}

template <size_t N, typename T>
MOCHI_FORCE_INLINE constexpr NdArray<T, N, N> Eye() {
  return DiagonalMatrix<N, T>((T)1);
}

// Return an NxN array with a given array down the diagonal.
template <size_t N, typename T>
MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, N> VDiagonalMatrix(Simd<T, 4> const& diagonalVector) {
  static_assert(N >= 1 && N <= 4, "Requires 1-4 rows");
  using V = Simd<T, 4>;
  NdArray<V, N> result = {};
  auto zeros = SimdZero<V>();
  if constexpr (N >= 1) {
    result[0] = Blend<0, 1, 1, 1>(diagonalVector, zeros);
  }
  if constexpr (N >= 2) {
    result[1] = Blend<1, 0, 1, 1>(diagonalVector, zeros);
  }
  if constexpr (N >= 3) {
    result[2] = Blend<1, 1, 0, 1>(diagonalVector, zeros);
  }
  if constexpr (N == 4) {
    result[3] = Blend<1, 1, 1, 0>(diagonalVector, zeros);
  }
  return result;
}

template <size_t N, typename T>
MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, N> VDiagonalMatrix(T valueOnDiagonal) {
  return VDiagonalMatrix<N>(Simd<T, 4>{valueOnDiagonal});
}

template <size_t N, typename T>
MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, N> VEye() {
  return VDiagonalMatrix<N, T>(T(1));
}

/**************************************************************************************************
  Matrix Determinant
*/

template <typename T>
MOCHI_FORCE_INLINE constexpr T Det(NdArray<T, 2, 2> const& A) {
  return A[0][0] * A[1][1] - A[0][1] * A[1][0];
}

template <typename T>
MOCHI_FORCE_INLINE constexpr T Det(NdArray<T, 3, 3> const& A) {
  return //
      A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
      A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
      A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
}

template <typename T>
MOCHI_FORCE_INLINE constexpr T Det(NdArray<T, 4, 4> const& A) {
  return Det4x4(&A[0][0]);
}

template <typename T, size_t N>
MOCHI_FORCE_INLINE Simd<T, 4> VDet3x3(NdArray<Simd<T, 4>, N> const& A) {
  static_assert(N == 3 || N == 4, "Expected a 3x3 or 4x4 Simd matrix");

  // Equivalent to:
  //
  // return
  //   A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
  //   A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
  //   A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
  //
  auto const a = Shuffle<1, 0, 0, 0>(A[1]);
  auto const b = Shuffle<2, 2, 1, 0>(A[2]);
  auto const c = Shuffle<2, 2, 1, 0>(A[1]);
  auto const d = Shuffle<1, 0, 0, 0>(A[2]);
  auto const result = A[0] * (a * b - c * d);
  return Broadcast<0>(result) - Broadcast<1>(result) + Broadcast<2>(result);
}

// Calculate the determinant of the 3x3 portion of a Vec4r[N]. Ignores the last column.
template <typename T, size_t N>
MOCHI_FORCE_INLINE T Det3x3(NdArray<Simd<T, 4>, N> const& A) {
  return Get<0>(VDet3x3(A));
}

template <typename T>
MOCHI_FORCE_INLINE T Det2x2(Simd<T, 4> const& A) {
  return A[0] * A[3] - A[1] * A[2];
}
/**************************************************************************************************
  Matrix Transpose
*/

template <typename T, size_t N, size_t M>
MOCHI_FORCE_INLINE constexpr NdArray<T, M, N> Transpose(NdArray<T, N, M> const& mat) {
  NdArray<T, M, N> result = {};
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      result[i][j] = mat[j][i];
    }
  }
  return result;
}

template <typename T>
MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 4> Transpose4x4(NdArray<Simd<T, 4>, 4> const& m) {
  // Adapted from _MM_TRANSPOSE4_PS
  auto const temp0 = Shuffle<0, 1, 0, 1>(m[0], m[1]);
  auto const temp2 = Shuffle<2, 3, 2, 3>(m[0], m[1]);
  auto const temp1 = Shuffle<0, 1, 0, 1>(m[2], m[3]);
  auto const temp3 = Shuffle<2, 3, 2, 3>(m[2], m[3]);
  return {
      Shuffle<0, 2, 0, 2>(temp0, temp1),
      Shuffle<1, 3, 1, 3>(temp0, temp1),
      Shuffle<0, 2, 0, 2>(temp2, temp3),
      Shuffle<1, 3, 1, 3>(temp2, temp3),
  };
}

template <typename T>
MOCHI_FORCE_INLINE Simd<T, 4> Transpose2x2(Simd<T, 4> const& m) {
  // Input is row-major: [a00, a01, a10, a11]
  // Output is row-major: [a00, a10, a01, a11]
  return Shuffle<0, 2, 1, 3>(m);
}

template <typename T, size_t D0>
MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 3> Transpose3x3(NdArray<Simd<T, 4>, D0> const& m) {
  static_assert(D0 == 3 || D0 == 4, "Expected 3x3 or 4x3 matrix");
  // Transpose the 3x3 portion. Fill the 4th column with m[2][3]
  auto const temp0 = Shuffle<0, 1, 0, 1>(m[0], m[1]);
  auto const temp1 = Shuffle<2, 3, 2, 3>(m[0], m[1]);
  return {
      Shuffle<0, 2, 0, 3>(temp0, m[2]),
      Shuffle<1, 3, 1, 3>(temp0, m[2]),
      Shuffle<0, 2, 2, 3>(temp1, m[2]),
  };
}

/**************************************************************************************************
  Matrix Cofactor
*/

template <typename T>
MOCHI_FORCE_INLINE constexpr NdArray<T, 2, 2> Cofactor(NdArray<T, 2, 2> const& mat) {
  NdArray<T, 2, 2> cofactor{
      NdArray<T, 2>{mat[1][1], -mat[1][0]}, NdArray<T, 2>{-mat[0][1], mat[0][0]}};
  return cofactor;
}

template <typename T>
inline constexpr NdArray<T, 3, 3> Cofactor(NdArray<T, 3, 3> const& mat) {
  NdArray<T, 3, 3> cofactor{
      NdArray<T, 3>{
          mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1],
          mat[1][2] * mat[2][0] - mat[1][0] * mat[2][2],
          mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]},
      NdArray<T, 3>{
          mat[0][2] * mat[2][1] - mat[0][1] * mat[2][2],
          mat[0][0] * mat[2][2] - mat[0][2] * mat[2][0],
          mat[0][1] * mat[2][0] - mat[0][0] * mat[2][1]},
      NdArray<T, 3>{
          mat[0][1] * mat[1][2] - mat[0][2] * mat[1][1],
          mat[0][2] * mat[1][0] - mat[0][0] * mat[1][2],
          mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]}};
  return cofactor;
}

// Calculate the cofactor matrix from the 3x3 portion of a Vec4r[3].
// The last column of the result should be ignored.
template <typename T>
inline NdArray<Simd<T, 4>, 3> Cofactor3x3(NdArray<Simd<T, 4>, 3> const& mat) {
  // Equivalent to Transpose(Invert(mat, 1_r)), which reduces to:
  //
  // return {
  //   NdArray<T, 3>{mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1],
  //                 mat[1][2] * mat[2][0] - mat[1][0] * mat[2][2],
  //                 mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]},
  //   NdArray<T, 3>{mat[0][2] * mat[2][1] - mat[0][1] * mat[2][2],
  //                 mat[0][0] * mat[2][2] - mat[0][2] * mat[2][0],
  //                 mat[0][1] * mat[2][0] - mat[0][0] * mat[2][1]},
  //   NdArray<T, 3>{mat[0][1] * mat[1][2] - mat[0][2] * mat[1][1],
  //                 mat[0][2] * mat[1][0] - mat[0][0] * mat[1][2],
  //                 mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]},
  // };

  // Prepare multiples for row0 of the result
  auto const a1 = Shuffle<1, 2, 2, 0>(mat[1], mat[1]);
  auto const b1 = Shuffle<2, 1, 0, 2>(mat[2], mat[2]);
  auto const a2 = mat[1]; // Shuffle<0, 1, 0, 0>(mat[1], mat[1]);
  auto const b2 = Shuffle<1, 0, 0, 0>(mat[2], mat[2]);

  // Prepare multiplies for row1 of the result
  auto const c1 = Shuffle<2, 1, 0, 2>(mat[0], mat[0]);
  auto const d1 = Shuffle<1, 2, 2, 0>(mat[2], mat[2]);
  auto const c2 = Shuffle<1, 0, 0, 0>(mat[0], mat[0]);
  auto const d2 = mat[2]; // Shuffle<0, 1, 0, 0>(mat[2], mat[2]);

  // Prepare multiples for row2 of the result
  auto const e1 = Shuffle<1, 2, 2, 0>(mat[0], mat[0]);
  auto const f1 = Shuffle<2, 1, 0, 2>(mat[1], mat[1]);
  auto const e2 = mat[0]; // Shuffle<0, 1, 0, 0>(mat[0], mat[0]);
  auto const f2 = Shuffle<1, 0, 0, 0>(mat[1], mat[1]);

  // Multiply cofactors
  auto const a1b1 = a1 * b1;
  auto const a2b2 = a2 * b2;
  auto const c1d1 = c1 * d1;
  auto const c2d2 = c2 * d2;
  auto const e1f1 = e1 * f1;
  auto const e2f2 = e2 * f2;

  // Shuffle results of multiples so they can be subtracted and returned.
  return {
      Shuffle<0, 2, 0, 0>(a1b1, a2b2) - Shuffle<1, 3, 1, 0>(a1b1, a2b2),
      Shuffle<0, 2, 0, 0>(c1d1, c2d2) - Shuffle<1, 3, 1, 0>(c1d1, c2d2),
      Shuffle<0, 2, 0, 0>(e1f1, e2f2) - Shuffle<1, 3, 1, 0>(e1f1, e2f2),
  };
}

// Calculate the cofactor matrix from the symmetric 2x2 SIMD matrix.
template <typename T>
MOCHI_FORCE_INLINE Simd<T, 4> CofactorSym2x2(Simd<T, 4> const& mat) {
  return Neg<false, true, false, false>(Shuffle<2, 1, 0, 3>(mat));
}

// Calculate the cofactor matrix from the symmetric 3x3 SIMD matrix.
template <typename T>
MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 2> CofactorSym3x3(NdArray<Simd<T, 4>, 2> const& mat) {
  // Equivalent to Transpose(Invert(mat, 1_r)), which reduces to:
  //
  //  |bc - ff, ef - dc, df - be|
  //  |   ·   , ac - ee, de - af|
  //  |   ·   ,    ·   , ab - dd|
  //
  // diag = (bc - ff, ac - ee, ab - dd)
  // offd = (ef - dc, df - be, de - af)

  auto diag = Shuffle<1, 0, 0, 3>(mat[0]) * Shuffle<2, 2, 1, 3>(mat[0]) -
      Shuffle<2, 1, 0, 3>(mat[1] * mat[1]);

  auto offd = Shuffle<1, 0, 0, 3>(mat[1]) * Shuffle<2, 2, 1, 3>(mat[1]) -
      mat[1] * Shuffle<2, 1, 0, 3>(mat[0]);

  return NdArray<Simd<T, 4>, 2>{diag, offd};
}

/**************************************************************************************************
  Matrix Trace (sum of diagonal elements)
*/
template <typename T, size_t DimTotal, size_t DimTrace>
MOCHI_FORCE_INLINE constexpr T Trace(NdArray<T, DimTotal, DimTotal> const& mat) {
  static_assert(DimTrace <= DimTotal, "Invalid dimensions");
  static_assert(DimTotal >= 1, "Invalid dimensions");
  static_assert(DimTrace >= 1, "Invalid dimensions");
  T trace = T(0);
  if constexpr (DimTrace >= 1) {
    trace += mat[0][0];
  }
  if constexpr (DimTrace >= 2) {
    trace += mat[1][1];
  }
  if constexpr (DimTrace >= 3) {
    trace += mat[2][2];
  }
  if constexpr (DimTrace >= 4) {
    trace += mat[3][3];
  }
  if constexpr (DimTrace >= 5) {
    for (int i = 4; i < DimTrace; ++i) {
      trace += mat[i][i];
    }
  }
  return trace;
}

template <typename T>
MOCHI_FORCE_INLINE constexpr T Trace2x2(Simd<T, 4> const& mat) {
  return mat[0] + mat[3];
}

template <typename T>
MOCHI_FORCE_INLINE constexpr T Trace3x3(NdArray<Simd<T, 4>, 3> const& mat) {
  return Get<0>(mat[0]) + Get<1>(mat[1]) + Get<2>(mat[2]);
}

/**************************************************************************************************
  Skew-symmetric matrix
*/

// Return the 3x3 skew-symmetric matrix [v] of a vector v, s.t. [v] * u = v x u, [v] = -[v]^T
template <typename T>
MOCHI_FORCE_INLINE constexpr NdArray<T, 3, 3> Skew(NdArray<T, 3> const& v) {
  T const zero{0};
  return {
      NdArray<T, 3>{zero, -v[2], v[1]},
      NdArray<T, 3>{v[2], zero, -v[0]},
      NdArray<T, 3>{-v[1], v[0], zero}};
}

// Return the 3x3 skew-symmetric matrix [v] of a vector v, s.t. [v] * u = v x u, [v] = -[v]^T
template <typename T>
MOCHI_FORCE_INLINE constexpr NdArray<Simd<T, 4>, 3> Skew3(Simd<T, 4> const& vector) {
  Simd<T, 4> v = Set<3>(vector, T(0));
  return {
      Neg<false, true, false, false>(Shuffle<3, 2, 1, 3>(v)), // {0, -v2, v1}
      Neg<false, false, true, false>(Shuffle<2, 3, 0, 3>(v)), // {v2, 0, -v0}
      Neg<true, false, false, false>(Shuffle<1, 0, 3, 3>(v)) // {-v1, v0, 0}
  };
}

// Return the vector v s.t. skew(v) is the anti-symmetric part of the input matrix
template <typename T>
MOCHI_FORCE_INLINE constexpr Simd<T, 4> InvSkew3(NdArray<Simd<T, 4>, 3> const& matrix) {
  return T(0.5) *
      Simd<T, 4>{
          matrix[2][1] - matrix[1][2], matrix[0][2] - matrix[2][0], matrix[1][0] - matrix[0][1]};
}

// Computes the first derivative of Skew(v)
template <typename T>
MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 3, 3> VDSkew3() {
  using V = Simd<T, 4>;
  NdArray<V, 3, 3> dskew;
  dskew[0] = Skew3(SimdBasisVector<0, V>());
  dskew[1] = Skew3(SimdBasisVector<1, V>());
  dskew[2] = Skew3(SimdBasisVector<2, V>());
  return dskew;
}

/**************************************************************************************************
  Vector Derivatives
*/

// Derivative of normalized 3-vector w.r.t. vector
template <typename T>
MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 3> DNormalize3(Simd<T, 4> const& v) {
  T const vNorm = Norm<3>(v) + std::numeric_limits<T>::min();
  Simd<T, 4> const vHat = v / vNorm;
  return (VEye<3, T>() - Outer3(vHat, vHat)) / vNorm;
}

template <typename T, size_t N>
MOCHI_FORCE_INLINE constexpr NdArray<T, N, N> DNormalize(NdArray<T, N> const& v) {
  return DNormalize(v, NormSqr(v));
}

template <typename T, size_t N>
MOCHI_FORCE_INLINE constexpr NdArray<T, N, N> DNormalize(NdArray<T, N> const& v, T sqrNorm) {
  constexpr auto kMin = std::numeric_limits<ScalarType<T>>::min();
  static_assert(kMin > 0); // Check numeric_limits has been correctly specialized.
  T const invNorm = T(1) / (Sqrt(sqrNorm) + T(kMin));
  NdArray<T, N> const n = v * invNorm;
  NdArray<T, N, N> result{};
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      result[i][j] = -n[i] * n[j] * invNorm;
    }
    result[i][i] += invNorm;
  }
  return result;
}

/**************************************************************************************************
  Matrix Row/Column Selection
*/

// Retrieves the largest row (in the L-2 sense) of the given matrix. Optionally outputs
// its squared norm.
template <typename T, size_t D0, size_t D1>
inline constexpr auto LargestRow(NdArray<T, D0, D1> const& A, T* sqrNorm) {
  // Compute per-row norms.
  NdArray<T, D0> sqrNorms = {};
  for (size_t i = 0; i < D0; ++i) {
    sqrNorms[i] = NormSqr(A[i]);
  }

  // Identify row with largest squared L2 norm.
  auto index = ArgMax(sqrNorms);

  // Store its squared norm if requested.
  if (sqrNorm != nullptr) {
    *sqrNorm = sqrNorms[index];
  }

  // Return largest row.
  return A[index];
}

inline auto LargestRowColSym2x2(VSymMatrix2x2r A, Vec4r& outNormSqr) {
  // Vectors are:
  // r₁ = (a, c)
  // r₂ = (c, b)

  // Evaluate squared norms.
  Vec4r entriesSqr = A * A; // (a², c², b², ?)
  Vec4r temp0 = Broadcast<1>(entriesSqr); // (c², c², ?, ?)
  Vec4r normSqr = entriesSqr + temp0; // (|r₁|², ?, |r₂|², ?)
  real r1NormSqr = Get<0>(normSqr);
  real r2NormSqr = Get<2>(normSqr);

  // Shuffle values according depending on which row was larger
  if (r1NormSqr > r2NormSqr) {
    outNormSqr = r1NormSqr;
    return Shuffle<0, 1, 3, 3>(A);
  } else {
    outNormSqr = r2NormSqr;
    return Shuffle<1, 2, 3, 3>(A);
  }
}

inline auto LargestRowColSym3x3(VSymMatrix3x3r A, Vec4r& outNormSqr) {
  // Vectors are:
  // r₁ = (a, d, e)
  // r₂ = (d, b, f)
  // r₃ = (e, f, c)

  // Evaluate squared norms.
  Vec4r diagSqr = A[0] * A[0]; // (a², b², c², ?)
  Vec4r offdSqr = A[1] * A[1]; // (d², e², f², ?)
  Vec4r temp0 = Shuffle<0, 0, 2, 3>(offdSqr); // (d², d², f², ?)
  Vec4r temp1 = Shuffle<1, 2, 1, 3>(offdSqr); // (e², f², e², ?)
  Vec4r normSqr = diagSqr + temp0 + temp1; // (|r₁|², |r₂|², |r₃|², ?)

  // Blend entries according to norms.
  real max = HMax<3>(normSqr);
  if (max == Get<0>(normSqr)) { // If r1 was the largest row
    outNormSqr = Broadcast<0>(normSqr);
    return Blend<0, 1, 1, 0>(A[0], Shuffle<0, 0, 1, 0>(A[1]));
  } else if (max == Get<1>(normSqr)) { // If r2 was the largest row
    outNormSqr = Broadcast<1>(normSqr);
    return Blend<1, 0, 1, 0>(A[0], Shuffle<0, 0, 2, 0>(A[1]));
  } else { // If r3 was the largest row
    outNormSqr = Broadcast<2>(normSqr);
    return Blend<1, 1, 0, 0>(A[0], Shuffle<1, 2, 0, 0>(A[1]));
  }
}

} // namespace mochi
