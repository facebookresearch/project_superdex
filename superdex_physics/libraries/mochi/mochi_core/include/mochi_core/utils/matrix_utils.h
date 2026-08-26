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

#include <mochi_core/mochi_config.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/vmatrix.h>

namespace mochi {

/**************************************************************************************************
  Frobenius Norm
*/

// Square of the Frobenius norm of a matrix
template <typename T, size_t N, size_t M>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr T NormSqr(NdArray<T, N, M> const& a);

// Frobenius norm of a matrix
template <typename T, size_t N, size_t M>
[[nodiscard]] MOCHI_FORCE_INLINE T Norm(NdArray<T, N, M> const& a);

// Square of the Frobenius norm of the upper-left 3x3 portion of a SIMD matrix
template <typename T, size_t D0, int D1, MOCHI_CONCEPT(D0 >= 3 && D1 >= 3)>
[[nodiscard]] MOCHI_FORCE_INLINE T NormSqr3x3(NdArray<Simd<T, D1>, D0> const& a);

// Frobenius norm of the upper-left 3x3 portion of a SIMD matrix.
template <typename T, size_t D0, int D1, MOCHI_CONCEPT(D0 >= 3 && D1 >= 3)>
[[nodiscard]] MOCHI_FORCE_INLINE T Norm3x3(NdArray<Simd<T, D1>, D0> const& a);

/**************************************************************************************************
  Matrix-Vector Dot Product
*/

// Dot product of (matrix, array). The result is a 1D array where the ith element is defined as the
// dot product of 'b' and a row (2nd dimension) of 'a'.
template <typename T, size_t N, size_t M>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr NdArray<T, N> DotMatVec(
    NdArray<T, N, M> const& a,
    NdArray<T, M> const& b);

template <typename T, size_t N, size_t M>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr NdArray<T, N> DotMatVec(
    NdArray<T, N, M> const& a,
    Span<T const> b);

template <typename T, size_t N, size_t M>
MOCHI_FORCE_INLINE void DotMatVec(NdArray<T, N, M> const& a, Span<T const> b, Span<T> out);

// Dot product of a 3xN matrix and an Nx1 vector, where N <= 4. The result is interpreted as a 3x1
// vector, with an additional invalid component for SIMD padding.  The matrix is allowed to have
// more (D0) rows, but rows beyond the third one will be ignored.
template <size_t N, typename T, size_t D0>
[[nodiscard]] MOCHI_FORCE_INLINE Simd<T, 4> DotMatVec3xN(
    NdArray<Simd<T, 4>, D0> const& m,
    Simd<T, 4> v);

// Dot product of (matrix, array). Supports 3x3 and 4x4 matrices, but only uses the upper-left 3x3
// portion of the matrix. Ignores v[3]. Can be used to rotate 3D vectors, but DotVecMat3x3 is faster
// if you already have the matrix transpose.
template <typename T, size_t D0>
[[nodiscard]] MOCHI_FORCE_INLINE Simd<T, 4> DotMatVec3x3(
    NdArray<Simd<T, 4>, D0> const& m,
    Simd<T, 4> v);

// Dot product of (matrix, array). Can be used to transform vectors, but DotVecMat4x4 is faster if
// you already have the matrix transpose.
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE Simd<T, 4> DotMatVec4x4(
    NdArray<Simd<T, 4>, 4> const& m,
    Simd<T, 4> v);

/**************************************************************************************************
  Vector-Matrix Dot Product
*/

// Dot product of (array, matrix). The result is a 1D array where the ith element is defined as the
// dot product of 'a' and a column (1st dimension) of 'b'.
template <typename T, size_t N, size_t M>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr NdArray<T, M> DotVecMat(
    NdArray<T, N> const& a,
    NdArray<T, N, M> const& b);

// Dot product of (array, matrix). SIMD specialization using the first 2 components of each Vec4r.
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE Simd<T, 4> DotVecMat2x3(
    Simd<T, 4> v,
    NdArray<Simd<T, 4>, 2> const& m);

// Dot product of (array, matrix). Input matrices can be VMatrix3x3r or VMatrix4x4r, but only the
// upper-left 3x3 portion will be used. Output is a Vec4r where the last SIMD component is
// undefined.
template <typename T, size_t D0>
[[nodiscard]] MOCHI_FORCE_INLINE Simd<T, 4> DotVecMat3x3(
    Simd<T, 4> v,
    NdArray<Simd<T, 4>, D0> const& m);

// Dot product of (array, matrix). Full SIMD specialization.
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE Simd<T, 4> DotVecMat4x4(
    Simd<T, 4> a,
    NdArray<Simd<T, 4>, 4> const& b);

/**************************************************************************************************
  Matrix-Matrix Dot Product
*/

// Dot product of 2 matrices
template <typename T, size_t N, size_t M, size_t L>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr NdArray<T, N, L> Dot(
    NdArray<T, N, M> const& a,
    NdArray<T, M, L> const& B);

// Dot product of 2 matrices. VMatrix4x4r specialization.
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 4> Dot4x4(
    NdArray<Simd<T, 4>, 4> const& a,
    NdArray<Simd<T, 4>, 4> const& b);

// Dot product of (matrix, matrix). Input matrices can be VMatrix3x3r or VMatrix4x4r, but only the
// upper-left 3x3 portion will be used. Output is a VMatrix3x3r where the last SIMD component of
// each row is undefined.
template <typename T, size_t D0A, size_t D0B>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 3> Dot3x3(
    NdArray<Simd<T, 4>, D0A> const& a,
    NdArray<Simd<T, 4>, D0B> const& b);

// Matrix-matrix product of 2x2 matrices.
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE Simd<T, 4> Dot2x2(Simd<T, 4> const& a, Simd<T, 4> const& b);

/**************************************************************************************************
  Outer Product
*/

// Outer product of two 2x1 vectors returning a 2x2 matrix
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr NdArray<T, 2, 2> Outer(
    NdArray<T, 2> const& a,
    NdArray<T, 2> const& b);

// Outer product of two vectors, 3x1 and 1x2, returning a 3x2 matrix
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr NdArray<T, 3, 2> Outer(
    NdArray<T, 3> const& a,
    NdArray<T, 2> const& b);

// Outer product of two 3x1 vectors returning a 3x3 matrix
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr NdArray<T, 3, 3> Outer(
    NdArray<T, 3> const& a,
    NdArray<T, 3> const& b);

// Outer product of two 3x1 vectors returning a 3x3 matrix
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 3> Outer3(Simd<T, 4> a, Simd<T, 4> b);

// Outer product of a 3x1 vector and a 3x3 matrix returning a 3x3x3 3rd-order tensor
template <typename T>
[[nodiscard]] NdArray<Simd<T, 4>, 3, 3> Outer3(
    Simd<T, 4> const& vec,
    NdArray<Simd<T, 4>, 3> const& mat);

// Outer product of a 3x3 matrix and a 3x1 vector returning a 3x3x3 3rd-order tensor
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 3, 3> Outer3(
    NdArray<Simd<T, 4>, 3> const& mat,
    Simd<T, 4> const& vec);

// Outer product of a 3x3x3 tensor and a 3x1 vector returning a 3x3x3x3 4th-order tensor
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 3, 3, 3> Outer3(
    NdArray<Simd<T, 4>, 3, 3> const& ten,
    Simd<T, 4> const& vec);

// Outer product of a 3x1 vector and 3x3x3 tensor and returning a 3x3x3x3 4th-order tensor
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 3, 3, 3> Outer3(
    Simd<T, 4> const& vec,
    NdArray<Simd<T, 4>, 3, 3> const& ten);

// Outer product of two 3x3 matrices returning a 3x3x3x3 4th-order tensor
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 3, 3, 3> Outer3(
    NdArray<Simd<T, 4>, 3> const& mat0,
    NdArray<Simd<T, 4>, 3> const& mat1);

// Outer product of 3 component vectors, returning a symmetric 3x3 SIMD matrix (assumed 4th
// component unused).
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 2> OuterSym3(Simd<T, 4> a, Simd<T, 4> b);

// Outer product of two 2x2 matrices, each stored in row-major order as SIMD vectors, returning a
// 2x2x2x2 4th-order tensor whose last two indices are packed into a single SIMD vector, with
// row-major order.  For `c = Outer2(a, b)`, we have `c[i][j][2*k + l] = a[2*i + j] * b[2*k + l]`,
// corresponding to  $c_{ijkl} = a_{ij} b_{kl}$ in mathematical index notation.
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 2, 2> Outer2(
    Simd<T, 4> const& a,
    Simd<T, 4> const& b);

/**************************************************************************************************
  Inner Product
*/

// Colon product (Frobenius inner product) of two 3x3 SIMD matrices A and B, i.e. the sum of the
// element-wise products A:B = \sum_{i,j} A_{ij} B_{ij}.
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE T
Colon3x3(NdArray<Simd<T, 4>, 3> const& A, NdArray<Simd<T, 4>, 3> const& B);

// Colon product (Frobenius inner product) of two matrices A and B, i.e. the sum of the element-wise
// products A:B = \sum_{i,j} A_{ij} B_{ij}.
template <typename T, size_t N, size_t M>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr T Colon(
    NdArray<T, N, M> const& A,
    NdArray<T, N, M> const& B);

// Colon product (Frobenius inner product) of two symmetric 2x2 matrices stored as raw
// [a00, a01, a11] components.
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr T ColonSym2x2(
    NdArray<T, 3> const& A,
    NdArray<T, 3> const& B);

/**************************************************************************************************
  Invert
*/

// Invert 2x2 matrix
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr NdArray<T, 2, 2> Invert(NdArray<T, 2, 2> const& a);

// Invert 2x2 matrix when the determinant is known
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr NdArray<T, 2, 2> Invert(
    NdArray<T, 2, 2> const& a,
    T det);

// Invert 3x3 matrix
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr NdArray<T, 3, 3> Invert(NdArray<T, 3, 3> const& a);

// Invert 3x3 matrix when the determinant is known
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr NdArray<T, 3, 3> Invert(
    NdArray<T, 3, 3> const& a,
    T det);

// The left pseudoinverse of a 3x2 matrix
template <typename T>
void PseudoInvert(NdArray<T, 3, 2> const& a, NdArray<T, 2, 3>* outInv, T* outDet);

// Invert a 4x4 matrix
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr NdArray<T, 4, 4> Invert(NdArray<T, 4, 4> const& A);

// Invert a 4x4 SIMD matrix
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr NdArray<Simd<T, 4>, 4> Invert4x4(
    NdArray<Simd<T, 4>, 4> const& A);

// Invert 3x3 SIMD matrix
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 3> Invert3x3(
    NdArray<Simd<T, 4>, 3> const& mat);

// Invert 3x3 SIMD matrix when the determinant is known
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 3> Invert3x3(
    NdArray<Simd<T, 4>, 3> const& mat,
    Simd<T, 4> det);

// Invert 2x2 SIMD matrix when the determinant is known
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE Simd<T, 4> Invert2x2(Simd<T, 4> const& mat, T det);

// Invert a 3D transformation matrix consisting of scale, rotation, and translation only.
// The matrix should be of the form:
//
//  | Ax Bx Cx Tx |    Where the first 3 columns are the orthogonal scaled basis vectors, and
//  | Ay By Cy Ty |    the last column is the translation
//  | Az Bz Cz Tz |
//  |  0  0  0  1 |
//
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 4> InvertTransformation(
    NdArray<Simd<T, 4>, 4> const& mat);

// Invert a pre-transposed 3D transformation matrix consisting of scale, rotation, and translation
// only. The input matrix should be of the form below. The result matrix will have the same form.
//
//  | Ax Ay Az 0 |     Where the first 3 rows are the orthogonal scaled basis vectors
//  | Bx By Bz 0 |     The last row is the translation
//  | Cx Cy Cz 0 |
//  | Tx Ty Tz 1 |
//
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 4> InvertTransformationTransposed(
    NdArray<Simd<T, 4>, 4> const& mat);

/**************************************************************************************************
  Matrix Builders
*/

// Return an NxN array with a given value down the diagonal. Equivalent to: (Eye<N, T>() *
// valueOnDiagonal)
template <size_t N, typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr NdArray<T, N, N> DiagonalMatrix(T valueOnDiagonal);

// Return an NxN array with a given array down the diagonal.
template <size_t N, typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr NdArray<T, N, N> DiagonalMatrix(
    NdArray<T, N> const& diagonalVector);

// Return raw [a00, a01, a11] components of a symmetric 2x2 matrix.
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr NdArray<T, 3> Sym2x2Components(T a00, T a01, T a11);

// Return raw [a00, a01, a11] components of the symmetric part of a 2x2 matrix.
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr NdArray<T, 3> Sym2x2Components(
    NdArray<T, 2, 2> const& m);

// Construct a 2x2 symmetric matrix from upper-triangle components.
//   [a00 a01]
//   [a01 a11]
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr NdArray<T, 2, 2> SymMatrix2x2(T a00, T a01, T a11);

// Return an NxN array with 1s down the diagonal (an identity matrix).
template <size_t N, typename T = real>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr NdArray<T, N, N> Eye();

// Return an NxN array with a given value down the diagonal. Equivalent to: (Eye<N, T>() *
// valueOnDiagonal)
template <size_t N, typename T>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, N> VDiagonalMatrix(T valueOnDiagonal);

// Return an NxN array with a given array down the diagonal.
template <size_t N, typename T>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, N> VDiagonalMatrix(
    Simd<T, 4> const& diagonalVector);

// Return an NxN array with 1s down the diagonal (an identity matrix).
template <size_t N, typename T = real>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, N> VEye();

/**************************************************************************************************
  Matrix Determinant
*/

// Determinant of 2x2 matrix.
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr T Det(NdArray<T, 2, 2> const& A);

// Determinant of 3x3 matrix.
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr T Det(NdArray<T, 3, 3> const& A);

// Determinant of the upper-left 3x3 of a SIMD matrix.
template <typename T, size_t N>
[[nodiscard]] MOCHI_FORCE_INLINE T
Det3x3(NdArray<Simd<T, 4>, N> const& A); // Ignores the last SIMD column

// Same as Det3x3 except that it returns Simd<T, 4>{det, det, det, det}
template <typename T, size_t N>
[[nodiscard]] MOCHI_FORCE_INLINE Simd<T, 4> VDet3x3(NdArray<Simd<T, 4>, N> const& A);

template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE T Det2x2(Simd<T, 4> const& A);

/**************************************************************************************************
  Matrix Transpose
*/

// Transpose matrix NxM
template <typename T, size_t N, size_t M>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr NdArray<T, M, N> Transpose(NdArray<T, N, M> const& mat);

// Transpose matrix 4x4 (SIMD)
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 4> Transpose4x4(
    NdArray<Simd<T, 4>, 4> const& m);

// Transpose a 2x2 SIMD matrix stored in row-major order as a single Simd<T, 4>.
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE Simd<T, 4> Transpose2x2(Simd<T, 4> const& m);

// Transpose the upper-left 3x3 portion of a SIMD matrix. The 4th column will be filled by m[2][3].
// Thus, if the 4th column was all zeros, then the 4th column of the transpose will also be all
// zeros. Input matrix can have have 3 or 4 rows.
template <typename T, size_t D0>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 3> Transpose3x3(
    NdArray<Simd<T, 4>, D0> const& m);

/**************************************************************************************************
  Matrix Cofactor
*/

template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr NdArray<T, 2, 2> Cofactor(NdArray<T, 2, 2> const& mat);

template <typename T>
[[nodiscard]] inline constexpr NdArray<T, 3, 3> Cofactor(NdArray<T, 3, 3> const& mat);

template <typename T>
[[nodiscard]] NdArray<Simd<T, 4>, 3> Cofactor3x3(
    NdArray<Simd<T, 4>, 3> const& mat); // Ignores the last SIMD column

template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE Simd<T, 4> CofactorSym2x2(Simd<T, 4> const& mat);

template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 2> CofactorSym3x3(
    NdArray<Simd<T, 4>, 2> const& mat);

/**************************************************************************************************
  Matrix Trace (sum of diagonal elements)
*/
template <typename T, size_t DimTotal = 3, size_t DimTrace = DimTotal>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr T Trace(NdArray<T, DimTotal, DimTotal> const& mat);

template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr T Trace2x2(Simd<T, 4> const& mat);

template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr T Trace3x3(NdArray<Simd<T, 4>, 3> const& mat);

/**************************************************************************************************
  Skew-symmetric matrix
*/

// Return the 3x3 skew-symmetric matrix [v] of a vector v, s.t. [v] * u = v x u, [v] = -[v]^T.
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr NdArray<T, 3, 3> Skew(NdArray<T, 3> const& v);

// Return the 3x3 skew-symmetric matrix [v] of a vector v, s.t. [v] * u = v x u, [v] = -[v]^T.
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr NdArray<Simd<T, 4>, 3> Skew3(Simd<T, 4> const& vector);

// Return the vector v s.t. skew(v) is the anti-symmetric part of the input matrix
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr Simd<T, 4> InvSkew3(
    NdArray<Simd<T, 4>, 3> const& matrix);

// Computes the first derivative of Skew(v)
template <typename T = real>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 3, 3> VDSkew3();

/**************************************************************************************************
  Vector Derivatives
*/

// Derivative of normalized 3-vector w.r.t. vector. Returns a 3x3 matrix representing dv_hat/dv,
// where v_hat = v / ||v||.
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 3> DNormalize3(Simd<T, 4> const& v);

// Derivative of normalized N-vector w.r.t. vector. Returns an NxN matrix representing dv_hat/dv,
// where v_hat = v / ||v||.
template <typename T, size_t N>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr NdArray<T, N, N> DNormalize(NdArray<T, N> const& v);

// Derivative of normalized N-vector w.r.t. vector with precomputed squared norm. Returns an NxN
// matrix representing dv_hat/dv, where v_hat = v / ||v||.
template <typename T, size_t N>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr NdArray<T, N, N> DNormalize(
    NdArray<T, N> const& v,
    T sqrNorm);

/**************************************************************************************************
  Matrix Row/Column Selection
*/

// Retrieves the largest row (in the L-2 sense) of the given matrix. Optionally outputs
// its squared norm.
template <typename T, size_t D0, size_t D1>
[[nodiscard]] constexpr auto LargestRow(NdArray<T, D0, D1> const& A, T* sqrNorm = nullptr);

// Retrieves the largest row/col (in the L-2 sense) of the given symmetric matrix.
// Optionally outputs its squared norm.
[[nodiscard]] auto LargestRowColSym2x2(VSymMatrix2x2r A, Vec4r& outNormSqr);

// Retrieves the largest row/col (in the L-2 sense) of the given symmetric matrix.
// Optionally outputs its squared norm.
[[nodiscard]] auto LargestRowColSym3x3(VSymMatrix3x3r A, Vec4r& outNormSqr);

} // namespace mochi

#include "matrix_utils_inl.h"
