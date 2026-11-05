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

#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/decomposition_utils.h>
#include <mochi_core/utils/matrix_utils.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace mochi {

/**************************************************************************************************
  Eigendecomposition of Symmetric Matrices: Core

  Shared by the scalar, SIMD, and batched solvers. `V` is `real`, or a `Simd` type holding one
  matrix per lane. A symmetric matrix is stored as its diagonal followed by its upper off-diagonal
  entries: [a, b, c] for the 2x2 matrix
  | a c |
  | c b |
  and [a, b, c, d, e, f] for the 3x3 matrix
  | a d e |
  | d b f |
  | e f c |
  and, except for SymScale and NormalizeSym, must be normalized to entries of order one.

  With eigenvectors, the decompositions are backward stable for every spectrum, including exactly or
  nearly repeated eigenvalues: an eigenvalue gap enters only ratios bounded by one and underflow
  guards. A Jacobi rotation diagonalizes a 2x2 matrix. For a 3x3 matrix, the eigenvector of the
  eigenvalue farthest from the other two comes from the cofactors of A - λI, then a Jacobi rotation
  diagonalizes A restricted to its orthogonal complement.
  REFERENCE: D. Eberly, "A Robust Eigensolver for 3x3 Symmetric Matrices" (2014).
*/

namespace detail {

// The mean absolute entry of the symmetric matrix `sym`, clamped to [realmin, 1 / realmin] so that
// its reciprocal is a normal number. The upper bound also catches a sum that overflows, and divides
// any finite entry to at most 4. Scaling the entries before summing them would not prevent the
// overflow: MSVC's /fp:fast factors the scaling out of the sum.
template <typename V, size_t N, size_t... I>
[[nodiscard]] MOCHI_FORCE_INLINE V SymScale(NdArray<V, N> const& sym, std::index_sequence<I...>) {
  real constexpr kMin = std::numeric_limits<real>::min();
  return Clamp(V{1_r / N} * (... + Abs(sym[I])), V{kMin}, V{1_r / kMin});
}

// Divide the symmetric matrix `sym` by SymScale, and return that scale. Off-diagonal entries of
// magnitude at most ε, the rounding error of the normalized matrix, are zeroed, so that a matrix
// diagonal to within rounding gets the coordinate axes as eigenvectors rather than a basis set by
// its rounding noise. The folds keep the entries in registers, where some compilers keep a loop's
// array in memory.
template <typename V, size_t N, size_t... I>
[[nodiscard]] MOCHI_FORCE_INLINE V NormalizeSym(NdArray<V, N>& sym, std::index_sequence<I...> seq) {
  constexpr size_t kDim = N == 3 ? 2 : 3;
  V const scale = SymScale(sym, seq);
  V const invScale = V{1_r} / scale;
  V const tol{std::numeric_limits<real>::epsilon()};
  ((sym[I] = I < kDim ? sym[I] * invScale
                      : Select(Abs(sym[I] * invScale) <= tol, V{0_r}, sym[I] * invScale)),
   ...);
  return scale;
}

template <typename V, size_t N>
[[nodiscard]] MOCHI_FORCE_INLINE V NormalizeSym(NdArray<V, N>& sym) {
  static_assert(N == 3 || N == 6, "Expected a 2x2 or 3x3 symmetric matrix");
  return NormalizeSym(sym, std::make_index_sequence<N>{});
}

// Eigenvalues lambda0 >= lambda1 of the symmetric matrix | a c ; c b |, and the eigenvector (x, y)
// of lambda0, so that (-y, x) is the eigenvector of lambda1. The Jacobi rotation is accurate to
// O(ε) for any gap between the eigenvalues.
template <typename V>
MOCHI_FORCE_INLINE void
EigendecompSym2x2(V const& a, V const& b, V const& c, V& lambda0, V& lambda1, V& x, V& y) {
  V const halfDiff = V{0.5_r} * (a - b);
  V const radiusSqr = halfDiff * halfDiff + c * c;
  V const radius = Sqrt(radiusSqr);
  // tan = c / w and cos² = 1 / (1 + tan²) = w / (2 radius), with w = |halfDiff| + radius. Below
  // realmin / ε, radius² may have lost precision to subnormal terms or to their flushing to zero,
  // but then the matrix is a multiple of the identity to within √(realmin / ε), far below the
  // rounding error of normalized entries, and needs no rotation.
  auto const isScalar =
      radiusSqr < V{std::numeric_limits<real>::min() / std::numeric_limits<real>::epsilon()};
  V const w = Select(isScalar, V{2_r}, Abs(halfDiff) + radius);
  V const tanRot = Select(isScalar, V{0_r}, c) / w;
  V const cosRot = Sqrt(w / (V{2_r} * Select(isScalar, V{1_r}, radius)));
  V const sinRot = tanRot * cosRot;
  auto const aIsLarger = halfDiff >= V{0_r};
  x = Select(aIsLarger, cosRot, sinRot);
  y = Select(aIsLarger, sinRot, cosRot);
  lambda0 = Max(a, b) + tanRot * c;
  lambda1 = Min(a, b) - tanRot * c;
}

// Solve the characteristic cubic of `sym` for the eigenvalue farthest from the other two, to
// O(ε‖A‖) for every spectrum. With q = tr(A)/3, p² = tr((A - qI)²)/6 and
// cos 3θ = det(A - qI)/(2p³), the eigenvalues are q + 2p cos(θ + 2πk/3). The farthest one is
// q + signedTwoP * cosSep, with signedTwoP = 2p sign(cos 3θ) and
// cosSep = cos(acos(|cos 3θ|)/3) ∈ [√3/2, 1]: the largest eigenvalue if cos 3θ ≥ 0, else the
// smallest.
template <typename V>
MOCHI_FORCE_INLINE void
SolveEigenCubicSym3x3(NdArray<V, 6> const& sym, V& q, V& signedTwoP, V& cosSep) {
  V const& a = sym[0];
  V const& b = sym[1];
  V const& c = sym[2];
  V const& d = sym[3];
  V const& e = sym[4];
  V const& f = sym[5];

  // The diagonal of A - qI from differences of diagonal entries, which are exact for close entries,
  // so that clustered eigenvalues keep their relative accuracy.
  V const third = V{1_r / 3_r};
  V const aMinusB = a - b;
  V const bMinusC = b - c;
  V const cMinusA = c - a;
  V const b0 = third * (aMinusB - cMinusA);
  V const b1 = third * (bMinusC - aMinusB);
  V const b2 = third * (cMinusA - bMinusC);
  q = third * (a + b + c);
  V const pSqr = V{1_r / 6_r} * (b0 * b0 + b1 * b1 + b2 * b2 + V{2_r} * (d * d + e * e + f * f));
  V const p = Sqrt(pSqr);
  V const detB = b0 * (b1 * b2 - f * f) - d * (d * b2 - e * f) + e * (d * f - b1 * e);
  V const cos3Theta =
      Clamp(detB / Max(V{2_r} * p * pSqr, V{std::numeric_limits<real>::min()}), V{-1_r}, V{1_r});

  // cosSep, the largest root of 4c³ - 3c = |cos 3θ|, is ₂F₁(-1/3, 1/3; 1/2; t / 2) in
  // t = 1 - |cos 3θ|. Its Taylor coefficients after the first are negative, so its degree-7 Taylor
  // polynomial is an upper bound, within 5.1e-5 for t ∈ [0, 1], from which Newton's method
  // decreases monotonically to it, with errors below 5.1e-9 and 5.2e-17 after 1 and 2 steps.
  constexpr int kNewtonSteps = MOCHI_USE_DOUBLE_PRECISION ? 2 : 1;
  V const absCos3Theta = Abs(cos3Theta);
  V const t = V{1_r} - absCos3Theta;
  V const t2 = t * t;
  V const terms01 = V{1_r} - V{1_r / 9_r} * t;
  V const terms23 = V{4_r / 243_r} + V{28_r / 6561_r} * t;
  V const terms45 = V{80_r / 59049_r} + V{2288_r / 4782969_r} * t;
  V const terms67 = V{23296_r / 129140163_r} + V{82688_r / 1162261467_r} * t;
  cosSep = terms01 - t2 * (terms23 + t2 * (terms45 + t2 * terms67));
  for (int i = 0; i < kNewtonSteps; ++i) {
    V const cosSepSqr = cosSep * cosSep;
    cosSep -=
        (cosSep * (V{4_r} * cosSepSqr - V{3_r}) - absCos3Theta) / (V{12_r} * cosSepSqr - V{3_r});
  }
  signedTwoP = V{2_r} * Sign(cos3Theta) * p;
}

// Eigenvalues of `sym` in descending order. The one farthest from the other two is accurate to
// O(ε‖A‖), but the other two are split by the discriminant of the cubic, which cancels as they
// merge: their error grows to O(√ε‖A‖) as their gap goes to zero. EigendecompSym3x3 has no such
// limitation.
template <typename V>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<V, 3> EigenvalsSym3x3(NdArray<V, 6> const& sym) {
  V q MOCHI_NO_INIT;
  V signedTwoP MOCHI_NO_INIT;
  V cosSep MOCHI_NO_INIT;
  SolveEigenCubicSym3x3(sym, q, signedTwoP, cosSep);

  // The other roots of 4c³ - 3c = |cos 3θ| are (-cosSep ± √3 sinSep) / 2.
  V const sqrt3SinSep =
      V{2_r * kSqrt3Over2} * Sqrt(Max((V{1_r} - cosSep) * (V{1_r} + cosSep), V{0_r}));
  V const cosMid = V{0.5_r} * (sqrt3SinSep - cosSep);
  V const cosFar = V{-0.5_r} * (sqrt3SinSep + cosSep);
  auto const sepIsLargest = signedTwoP >= V{0_r};
  V eigval0 = q + signedTwoP * Select(sepIsLargest, cosSep, cosFar);
  V eigval1 = q + signedTwoP * cosMid;
  V eigval2 = q + signedTwoP * Select(sepIsLargest, cosFar, cosSep);

  // A diagonal `sym` returns its diagonal, which the caller's rescaling may round. Only the scalar
  // and SIMD solvers' diagonal fast path always returns a diagonal input exactly.
  auto const isDiagonal = Abs(sym[3]) + Abs(sym[4]) + Abs(sym[5]) <= V{0_r};
  eigval0 = Select(isDiagonal, sym[0], eigval0);
  eigval1 = Select(isDiagonal, sym[1], eigval1);
  eigval2 = Select(isDiagonal, sym[2], eigval2);

  // 3-element sorting network. Only the diagonal override can break the descending order.
  V const max01 = Max(eigval0, eigval1);
  V const min01 = Min(eigval0, eigval1);
  V const midTmp = Min(max01, eigval2);
  return {Max(max01, eigval2), Max(min01, midTmp), Min(min01, midTmp)};
}

// Largest row of the cofactor matrix of A - lambda I, and its squared norm.
template <typename V>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<V, 3>
LargestCofactorRowSym3x3(NdArray<V, 6> const& sym, V const& lambda, V& outNormSqr) {
  V const& d = sym[3];
  V const& e = sym[4];
  V const& f = sym[5];
  V const m00 = sym[0] - lambda;
  V const m11 = sym[1] - lambda;
  V const m22 = sym[2] - lambda;

  V const cof00 = m11 * m22 - f * f;
  V const cof01 = e * f - d * m22;
  V const cof02 = d * f - m11 * e;
  V const cof11 = m00 * m22 - e * e;
  V const cof12 = d * e - m00 * f;
  V const cof22 = m00 * m11 - d * d;

  V const cof01Sqr = cof01 * cof01;
  V const cof02Sqr = cof02 * cof02;
  V const cof12Sqr = cof12 * cof12;
  V const norm0 = cof00 * cof00 + cof01Sqr + cof02Sqr;
  V const norm1 = cof01Sqr + cof11 * cof11 + cof12Sqr;
  V const norm2 = cof02Sqr + cof12Sqr + cof22 * cof22;

  auto const use1 = norm1 > norm0;
  V const best01 = Select(use1, norm1, norm0);
  V rX = Select(use1, cof01, cof00);
  V rY = Select(use1, cof11, cof01);
  V rZ = Select(use1, cof12, cof02);

  auto const use2 = norm2 > best01;
  outNormSqr = Select(use2, norm2, best01);
  rX = Select(use2, cof02, rX);
  rY = Select(use2, cof12, rY);
  rZ = Select(use2, cof22, rZ);
  return {rX, rY, rZ};
}

template <typename V>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<V, 3> DotSymVec3x3(
    NdArray<V, 6> const& sym,
    NdArray<V, 3> const& x) {
  return {
      sym[0] * x[0] + sym[3] * x[1] + sym[4] * x[2],
      sym[3] * x[0] + sym[1] * x[1] + sym[5] * x[2],
      sym[4] * x[0] + sym[5] * x[1] + sym[2] * x[2]};
}

// Eigenvalues of `sym` in descending order, and its eigenvectors as the rows of a rotation.
template <typename V>
MOCHI_FORCE_INLINE void
EigendecompSym3x3(NdArray<V, 6> const& sym, NdArray<V, 3>& eigvals, NdArray<V, 3, 3>& eigvecs) {
  using V3 = NdArray<V, 3>;
  V const zero{0_r};
  V const one{1_r};
  V const realMin{std::numeric_limits<real>::min()};

  V q MOCHI_NO_INIT;
  V signedTwoP MOCHI_NO_INIT;
  V cosSep MOCHI_NO_INIT;
  SolveEigenCubicSym3x3(sym, q, signedTwoP, cosSep);
  V const lambdaSep = q + signedTwoP * cosSep;
  auto const sepIsSmallest = signedTwoP < zero;

  // The rows of the cofactor matrix of A - lambdaSep I are parallel to the eigenvector of
  // lambdaSep. An error δ in lambdaSep tilts them by δ/gap towards eigenvectors a gap away, so the
  // residual of the largest row stays O(δ) however small the gaps are. The rows all vanish only if
  // A = lambdaSep I to within underflow, where any unit vector is an eigenvector.
  V rowNormSqr MOCHI_NO_INIT;
  V3 const row = LargestCofactorRowSym3x3(sym, lambdaSep, rowNormSqr);
  auto const isZeroRow = rowNormSqr < realMin;
  V3 const n = {
      Select(isZeroRow, one, row[0]),
      Select(isZeroRow, zero, row[1]),
      Select(isZeroRow, zero, row[2])};
  V const nNorm = Sqrt(Select(isZeroRow, one, rowNormSqr));

  // e0 = n / |n|, and the orthonormal basis (u, v) of its complement with u × v = e0, from a single
  // division: Duff et al.'s k = -1 / (sign(e0_z) + e0_z) is -|n| / s, with s = sign(n_z) |n| + n_z.
  // REFERENCE: Duff et al., "Building an Orthonormal Basis, Revisited" (2017).
  V const sign = Sign(n[2]);
  V const s = sign * nNorm + n[2];
  V const invNormS = one / (nNorm * s);
  V const invNorm = s * invNormS;
  V3 const e0 = {n[0] * invNorm, n[1] * invNorm, n[2] * invNorm};
  V const xyk = -n[0] * n[1] * invNormS;
  V3 const u = {one - sign * n[0] * n[0] * invNormS, sign * xyk, -sign * e0[0]};
  V3 const v = {xyk, sign - n[1] * n[1] * invNormS, -e0[1]};

  // Diagonalize A restricted to span(u, v).
  V3 const Au = DotSymVec3x3(sym, u);
  V3 const Av = DotSymVec3x3(sym, v);
  V lambda1 MOCHI_NO_INIT;
  V lambda2 MOCHI_NO_INIT;
  V x MOCHI_NO_INIT;
  V y MOCHI_NO_INIT;
  EigendecompSym2x2(Dot(u, Au), Dot(v, Av), Dot(v, Au), lambda1, lambda2, x, y);
  V3 const e1 = x * u + y * v;
  V3 const e2 = x * v - y * u;

  // The Rayleigh quotient of e0, clamped so that rounding cannot break the descending order.
  V const rayleigh = Dot(e0, DotSymVec3x3(sym, e0));
  V const lambda0 = Select(sepIsSmallest, Min(rayleigh, lambda2), Max(rayleigh, lambda1));

  // Both (e0, e1, e2) and (e1, e2, e0) are right-handed.
  auto const select = [&](V3 const& ifSmallest, V3 const& ifLargest) {
    return V3{
        Select(sepIsSmallest, ifSmallest[0], ifLargest[0]),
        Select(sepIsSmallest, ifSmallest[1], ifLargest[1]),
        Select(sepIsSmallest, ifSmallest[2], ifLargest[2])};
  };
  eigvals = {
      Select(sepIsSmallest, lambda1, lambda0),
      Select(sepIsSmallest, lambda2, lambda1),
      Select(sepIsSmallest, lambda0, lambda2)};
  eigvecs = {select(e1, e0), select(e2, e1), select(e0, e2)};
}

// The scalar and SIMD solvers' conversion and diagonal fast path, on matrices that need not be
// normalized.
[[nodiscard]] MOCHI_FORCE_INLINE Real6 ToSym6(Matrix3x3r const& mat) {
  return {mat[0][0], mat[1][1], mat[2][2], mat[0][1], mat[0][2], mat[1][2]};
}

[[nodiscard]] MOCHI_FORCE_INLINE Real6 ToSym6(VSymMatrix3x3r const& mat) {
  return {
      Get<0>(mat[0]),
      Get<1>(mat[0]),
      Get<2>(mat[0]),
      Get<0>(mat[1]),
      Get<1>(mat[1]),
      Get<2>(mat[1])};
}

// Whether `sym` is diagonal to within rounding: its off-diagonal entries are at most ε times
// SymScale, which NormalizeSym would zero.
[[nodiscard]] MOCHI_FORCE_INLINE bool IsDiagonalSym3x3(Real6 const& sym) {
  real const tol =
      std::numeric_limits<real>::epsilon() * SymScale(sym, std::make_index_sequence<6>{});
  return Max(Abs(sym[3]), Abs(sym[4]), Abs(sym[5])) <= tol;
}

// The indices of the diagonal entries of `sym` in descending order.
[[nodiscard]] MOCHI_FORCE_INLINE Int3 SortDiagonalSym3x3(Real6 const& sym) {
  Int3 order = {0, 1, 2};
  if (sym[order[0]] < sym[order[1]]) {
    std::swap(order[0], order[1]);
  }
  if (sym[order[1]] < sym[order[2]]) {
    std::swap(order[1], order[2]);
  }
  if (sym[order[0]] < sym[order[1]]) {
    std::swap(order[0], order[1]);
  }
  return order;
}

} // namespace detail

/**************************************************************************************************
  Eigendecomposition of Symmetric Matrices: Scalar
*/

inline void
AnalyticalEigendecompSym(Matrix2x2r mat, Real2& eigvalues, Matrix2x2r* eigvecs, bool transpose) {
  Real3 sym = {mat[0][0], mat[1][1], mat[0][1]};
  real const scale = detail::NormalizeSym(sym);
  real x = 0_r;
  real y = 0_r;
  detail::EigendecompSym2x2(sym[0], sym[1], sym[2], eigvalues[0], eigvalues[1], x, y);
  eigvalues *= scale;
  if (eigvecs != nullptr) {
    // Eigenvectors are stored as rows (𝐐ᵀ), or as columns (𝐐) if transpose is false.
    *eigvecs =
        transpose ? Matrix2x2r{Real2{x, y}, Real2{-y, x}} : Matrix2x2r{Real2{x, -y}, Real2{y, x}};
  }
}

namespace detail {

// The body of the scalar 3x3 entry point, which the scalar SVDs call too. Force-inlined, unlike the
// entry point, which compilers may leave out of line.
MOCHI_FORCE_INLINE void AnalyticalEigendecompSymImpl(
    Matrix3x3r const& mat,
    Real3& eigvalues,
    Matrix3x3r* eigvecs,
    bool transpose) {
  Real6 sym = detail::ToSym6(mat);
  // A diagonal matrix, such as FᵀF for a rotation F, takes a fast path: its sorted diagonal, with
  // the permuted coordinate axes as eigenvectors.
  if (detail::IsDiagonalSym3x3(sym)) {
    Int3 const order = detail::SortDiagonalSym3x3(sym);
    eigvalues = {sym[order[0]], sym[order[1]], sym[order[2]]};
    if (eigvecs != nullptr) {
      Matrix3x3r const axes = Eye<3>();
      *eigvecs = {axes[order[0]], axes[order[1]], Cross(axes[order[0]], axes[order[1]])};
    }
  } else {
    real const scale = detail::NormalizeSym(sym);
    if (eigvecs != nullptr) {
      detail::EigendecompSym3x3(sym, eigvalues, *eigvecs);
    } else {
      eigvalues = detail::EigenvalsSym3x3(sym);
    }
    eigvalues *= scale;
  }
  // Eigenvectors are stored as rows (𝐐ᵀ). If 𝐐 was requested, transpose back.
  if (eigvecs != nullptr && !transpose) {
    *eigvecs = Transpose(*eigvecs);
  }
}

} // namespace detail

inline void AnalyticalEigendecompSym(
    Matrix3x3r const& mat,
    Real3& eigvalues,
    Matrix3x3r* eigvecs,
    bool transpose) {
  detail::AnalyticalEigendecompSymImpl(mat, eigvalues, eigvecs, transpose);
}

/**************************************************************************************************
  Eigendecomposition of Symmetric Matrices: SIMD
*/

inline void
AnalyticalEigendecompSym2x2(VSymMatrix2x2r mat, Vec4r& eigvalues, VMatrix2x2r* eigvecs) {
  Real3 sym = {Get<0>(mat), Get<2>(mat), Get<1>(mat)};
  real const scale = detail::NormalizeSym(sym);
  real lambda0 = 0_r;
  real lambda1 = 0_r;
  real x = 0_r;
  real y = 0_r;
  detail::EigendecompSym2x2(sym[0], sym[1], sym[2], lambda0, lambda1, x, y);
  eigvalues = Vec4r{lambda0 * scale, lambda1 * scale, 0_r, 0_r};
  if (eigvecs != nullptr) {
    *eigvecs = Vec4r{x, y, -y, x};
  }
}

namespace detail {

// The body of the SIMD 3x3 entry point, which the SIMD SVDs and PSD projection call too.
// Force-inlined, unlike the entry point, which compilers may leave out of line.
MOCHI_FORCE_INLINE void
AnalyticalEigendecompSym3x3Impl(VSymMatrix3x3r const& mat, Vec4r& eigvalues, VMatrix3x3r* eigvecs) {
  Real6 sym = detail::ToSym6(mat);
  // A diagonal matrix, such as FᵀF for a rotation F, takes a fast path: its sorted diagonal, with
  // the permuted coordinate axes as eigenvectors.
  if (detail::IsDiagonalSym3x3(sym)) {
    Int3 const order = detail::SortDiagonalSym3x3(sym);
    eigvalues = Vec4r{sym[order[0]], sym[order[1]], sym[order[2]]};
    if (eigvecs != nullptr) {
      VMatrix3x3r const axes = VEye<3>();
      *eigvecs = {axes[order[0]], axes[order[1]], Cross3(axes[order[0]], axes[order[1]])};
    }
    return;
  }
  real const scale = detail::NormalizeSym(sym);
  Real3 vals MOCHI_NO_INIT;
  if (eigvecs != nullptr) {
    Matrix3x3r vecs MOCHI_NO_INIT;
    detail::EigendecompSym3x3(sym, vals, vecs);
    *eigvecs = ToSimdMatrix(vecs);
  } else {
    vals = detail::EigenvalsSym3x3(sym);
  }
  eigvalues = ToSimd(vals * scale);
}

} // namespace detail

inline void
AnalyticalEigendecompSym3x3(VSymMatrix3x3r const& mat, Vec4r& eigvalues, VMatrix3x3r* eigvecs) {
  detail::AnalyticalEigendecompSym3x3Impl(mat, eigvalues, eigvecs);
}

/**************************************************************************************************
  Singular-Value Decomposition: Scalar
*/

// WARNING: Results may be inaccurate if 𝐅ᵀ𝐅 or det(𝐅) overflows or underflows.
inline void RotationVariantSvdVals(Matrix3x3r const& F, Real3& Sg) {
  // Form normal matrix 𝐆 = 𝐅ᵀ𝐅. Perform eigendecomposition of 𝐆.
  Real3 eigvalues;
  detail::AnalyticalEigendecompSymImpl(Dot(Transpose(F), F), eigvalues, nullptr, false);

  // Compute singular values σᵢ² = λᵢ
  // NOTE: Although the eigenvalues should be in ℝ⁺, explicitly clamp them
  // before computing the square root in case numerical roundoff happened.
  Sg[0] = std::sqrt(std::max(0.0_r, eigvalues[0]));
  Sg[1] = std::sqrt(std::max(0.0_r, eigvalues[1]));
  Sg[2] = std::sqrt(std::max(0.0_r, eigvalues[2]));

  // If an inversion happened (i.e. det(𝐅) < 0), encode this information by
  // negating the smallest singular value.
  if (Det(F) < 0.0_r) {
    Sg[2] = -Sg[2];
  }
}

// WARNING: Results may be inaccurate if 𝐅ᵀ𝐅 or det(𝐅) overflows or underflows.
inline void RotationVariantSvd(Matrix3x3r const& F, Matrix3x3r& U, Real3& Sg, Matrix3x3r& VT) {
  // Form normal matrix 𝐆 = 𝐅ᵀ𝐅. Perform eigendecomposition of 𝐆.
  Real3 eigvalues;
  Matrix3x3r G = Dot(Transpose(F), F);
  detail::AnalyticalEigendecompSymImpl(G, eigvalues, &VT, true);

  // Compute singular values σᵢ² = λᵢ
  // NOTE: Although the eigenvalues should be in ℝ⁺, explicitly clamp them before
  // computing the square root in case numerical roundoff happened.
  Sg[0] = std::sqrt(std::max(0.0_r, eigvalues[0]));
  Sg[1] = std::sqrt(std::max(0.0_r, eigvalues[1]));
  Sg[2] = std::sqrt(std::max(0.0_r, eigvalues[2]));

  // If an inversion happened (i.e. det(𝐅) < 0), encode this information by
  // negating the smallest singular value.
  if (Det(F) < 0.0_r) {
    Sg[2] = -Sg[2];
  }

  // Compute singular vectors 𝐔=𝐅𝐕𝚺⁻¹
  real constexpr kEpsilon = std::numeric_limits<real>::epsilon();
  real constexpr kMin = std::numeric_limits<real>::min();
  real const invSigma0 = 1_r / (Sg[0] + kMin);

  // Compute first row of UT. It is unit to rounding, since σ₀² is the eigenvalue of 𝐆 for VT[0].
  if (Sg[0] * Sg[0] <= kMin) MOCHI_UNLIKELY {
    U[0] = Real3{1.0_r, 0.0_r, 0.0_r};
  } else {
    U[0] = DotMatVec(F, VT[0]) * invSigma0;
  }

  // Compute second row of UT: 𝐅VT[1]/σ₀ minus its component along U[0]. Rounding leaves that
  // component at about ε, which would tilt U[1] by about εσ₀/σ₁, and which the rank test below
  // would mistake for a nonzero σ₁ if 𝐅 is rank-deficient. Divided by σ₀, the test is relative,
  // and a U[1] that passes it has a squared norm of at least ε², far from underflow.
  U[1] = DotMatVec(F, VT[1]) * invSigma0;
  U[1] -= Dot(U[0], U[1]) * U[0];
  real const u1normSqr = NormSqr(U[1]);
  if (u1normSqr <= kEpsilon * kEpsilon) MOCHI_UNLIKELY {
    U[1] = Normalize(OrthogonalVector(U[0]));
  } else {
    U[1] /= Sqrt(u1normSqr);
  }

  // Compute third row of UT
  U[2] = Cross(U[0], U[1]);

  // Transpose to return U
  U = Transpose(U);
}

/**************************************************************************************************
  Singular-Value Decomposition: SIMD
*/

// WARNING: Results may be inaccurate if 𝐅ᵀ𝐅 or det(𝐅) overflows or underflows.
inline void RotationVariantSvdVals3x3(VMatrix3x3r const& F, Vec4r& Sg) {
  // Compute F^T*F
  VSymMatrix3x3r Gsym = SimdFullToSym(Dot3x3(Transpose3x3(F), F));

  Vec4r eigvalues;
  detail::AnalyticalEigendecompSym3x3Impl(Gsym, eigvalues, nullptr);

  // Compute the singular values (i.e. sqrt(eigvalues), clamp
  // them before computing sqrt in case there is rounding error
  // making them negative
  Sg = Sqrt(Max(eigvalues, Vec4r{0_r}));

  // If an inversion happened (i.e. det(𝐅) < 0), encode this
  // information by negating the smallest singular value.
  if (Det3x3(F) < 0_r) {
    Sg = Neg<0, 0, 1, 0>(Sg); // Negate flagged entries
  }
}

// WARNING: Results may be inaccurate if 𝐅ᵀ𝐅 or det(𝐅) overflows or underflows.
inline void
RotationVariantSvd3x3(VMatrix3x3r const& F, VMatrix3x3r& U, Vec4r& Sg, VMatrix3x3r& VT) {
  // Compute F^T * F
  VMatrix3x3r FT = Transpose3x3(F);
  VSymMatrix3x3r Gsym = SimdFullToSym(Dot3x3(FT, F));

  Vec4r eigvalues;
  detail::AnalyticalEigendecompSym3x3Impl(Gsym, eigvalues, &VT);

  // Compute the singular values (i.e. sqrt(eigvalues), clamp
  // them before computing sqrt in case there is rounding error
  // making them negative
  Sg = Sqrt(Max(eigvalues, Vec4r{0_r}));

  // If an inversion happened (i.e. det(𝐅) < 0), encode this
  // information by negating the smallest singular value.
  if (Det3x3(F) < 0_r) {
    Sg = Neg<0, 0, 1, 0>(Sg); // Negate flagged entries
  }

  // Compute singular vectors 𝐔=𝐅𝐕𝚺⁻¹.
  real constexpr kEpsilon = std::numeric_limits<real>::epsilon();
  real constexpr kMin = std::numeric_limits<real>::min();
  real const sigma0 = Get<0>(Sg);
  real const invSigma0 = 1_r / (sigma0 + kMin);

  // Compute first row of UT. It is unit to rounding, since σ₀² is the eigenvalue of 𝐆 for VT[0].
  if (sigma0 * sigma0 > kMin) MOCHI_LIKELY {
    U[0] = DotVecMat3x3(VT[0], FT) * invSigma0; // Faster than DotMatVec3x3
  } else MOCHI_UNLIKELY {
    U[0] = SimdBasisVector<0>();
  }

  // Compute second row of UT: 𝐅VT[1]/σ₀ minus its component along U[0]. Rounding leaves that
  // component at about ε, which would tilt U[1] by about εσ₀/σ₁, and which the rank test below
  // would mistake for a nonzero σ₁ if 𝐅 is rank-deficient. Divided by σ₀, the test is relative,
  // and a U[1] that passes it has a squared norm of at least ε², far from underflow.
  U[1] = DotVecMat3x3(VT[1], FT) * invSigma0; // Faster than DotMatVec3x3
  U[1] -= VDot<3>(U[0], U[1]) * U[0];
  real const u1normSqr = NormSqr<3>(U[1]);
  if (u1normSqr > kEpsilon * kEpsilon) MOCHI_LIKELY {
    U[1] = Normalize(U[1], u1normSqr);
  } else MOCHI_UNLIKELY {
    U[1] = Normalize<3>(OrthogonalVector3(U[0]));
  }

  // Compute third row of UT
  U[2] = Cross3(U[0], U[1]);

  // Transpose to return U
  U = Transpose3x3(U);
}

/**************************************************************************************************
  Polar Decomposition
*/

inline void LeftPolarDecomposition3x3(VMatrix3x3r const& A, VMatrix3x3r& U, VMatrix3x3r& P) {
  // Compute SVD of A.
  Vec4r sigma;
  VMatrix3x3r W, Vt;
  RotationVariantSvd3x3(A, W, sigma, Vt); // A = W * sigma * Vt

  // RotationVariantSvd3x3 encodes inversions by negating the smallest singular value. Revert this
  // convention.
  if (Get<2>(sigma) < 0_r) {
    Vt[2] = -Vt[2];
    sigma = Neg<0, 0, 1, 0>(sigma);
  }

  // Compute the left polar decomposition from the SVD.
  U = Dot3x3(W, Vt); // U = W * Vt
  P = Dot3x3(Dot3x3(Transpose3x3(Vt), VDiagonalMatrix<3>(sigma)), Vt); // P = V * sigma * Vt
}

/**************************************************************************************************
  Positive Semi-Definite Projection of Symmetric Matrices.
*/

namespace details {

/// @brief Cheap heuristic check for symmetric positive-definiteness (SPD) of a 3x3 matrix.
/// @details Uses Sylvester's strict leading-minor criterion: the three leading principal minors
/// must be strictly positive.
/// @note Assumes @p A is symmetric without verifying it.
[[nodiscard]] inline bool IsSpd(VMatrix3x3r const& A) {
  if (Get<0>(A[0]) <= 0_r) {
    return false;
  } else if ((Get<0>(A[0]) * Get<1>(A[1]) - Get<0>(A[1]) * Get<1>(A[0])) <= 0_r) {
    return false;
  } else if (Det3x3(A) <= 0_r) {
    return false;
  } else {
    return true;
  }
}

} // namespace details

inline void ProjectSymPsd(VMatrix2x2r& A, real eps) {
  MOCHI_ASSERT_VERBOSE(eps >= 0_r, "Negative eps.");

  // Perform eigendecomposition: A = Q * D * Q^T.
  // AnalyticalEigendecompSym2x2 returns eigenvectors as ROWS (i.e., Q^T).
  // Row-major storage: (q₀₀, q₀₁, q₁₀, q₁₁) = [[q₀₀, q₀₁], [q₁₀, q₁₁]].
  // Row 0 = eigenvector 0 = (q₀₀, q₀₁), Row 1 = eigenvector 1 = (q₁₀, q₁₁).
  Vec4r L;
  VMatrix2x2r QT;
  AnalyticalEigendecompSym2x2(SimdFullToSym(A), L, &QT);

  // Project eigenvalues to be non-negative.
  L = Max(L, Vec4r{eps});

  // Reconstruct A = Q * D * Q^T = (Q^T)^T * D * Q^T.
  // D * Q^T: scale each row of Q^T by its eigenvalue.
  // Storage: (q₀₀, q₀₁, q₁₀, q₁₁) * (λ₀, λ₀, λ₁, λ₁) = (λ₀*q₀₀, λ₀*q₀₁, λ₁*q₁₀, λ₁*q₁₁).
  VMatrix2x2r const DQT = QT * Shuffle<0, 0, 1, 1>(L);

  // Q = (Q^T)^T: transpose. Shuffle<0, 2, 1, 3> swaps elements 1 and 2.
  // (q₀₀, q₀₁, q₁₀, q₁₁) → (q₀₀, q₁₀, q₀₁, q₁₁).
  VMatrix2x2r const Q = Transpose2x2(QT);

  // Final result: Q * (D * Q^T).
  A = Dot2x2(Q, DQT);
}

inline void ProjectSymPsd(VMatrix3x3r& A, real eps) {
  MOCHI_ASSERT_VERBOSE(eps >= 0_r, "Negative eps.");
  // To improve performance and avoid accumulation of finite precision errors, project only if not
  // already positive definite. The fast-path predicate is intentionally a strict-Sylvester PD check
  // rather than a full PSD check: it requires only the three leading principal minors (cheap)
  // instead of all seven principal minors needed for a PSD test. Singular PSD matrices fall through
  // to the eigendecomposition and are projected correctly.
  if (!details::IsSpd(A)) {
    // Perform eigendecomposition.
    Vec4r L;
    VMatrix3x3r QT;
    detail::AnalyticalEigendecompSym3x3Impl(SimdFullToSym(A), L, &QT);

    // Perform projection.
    A = Dot3x3(Transpose3x3(QT), Dot3x3(VDiagonalMatrix<3>(Max(L, Vec4r{eps})), QT));
  }
}

inline VMatrix2x2r CholeskySym2x2(VMatrix2x2r const& A, real eps) {
  MOCHI_ASSERT_VERBOSE(eps > 0_r, "Non-positive eps.");

  // A is stored as (a00, a01, a10, a11) in row-major. For symmetric input, a01 == a10.
  real const a00 = Get<0>(A);
  real const a01 = Get<1>(A);
  real const a11 = Get<3>(A);

  // L00 = sqrt(max(a00, eps))
  real const l00 = Sqrt(Max(a00, eps));

  // L10 = a01 / L00 (off-diagonal, safe due to L00 >= sqrt(eps))
  real const l10 = a01 / l00;

  // L11 = sqrt(max(a11 - L10^2, eps))
  real const l11 = Sqrt(Max(a11 - l10 * l10, eps));

  // Return L in row-major format: (l00, 0, l10, l11)
  return VMatrix2x2r{l00, 0_r, l10, l11};
}

inline VMatrix2x2r ProjectPsdWithMetric(
    VMatrix2x2r const& S,
    VMatrix2x2r const& M,
    real epsCholesky,
    real epsEigenvalue) {
  // Cholesky factor of M
  VMatrix2x2r const L = CholeskySym2x2(M, epsCholesky);

  // Analytical inverse of lower-triangular L = (l00, 0, l10, l11).
  // L^(-1) = (1/l00, 0, -l10/(l00*l11), 1/l11).
  real const l00 = Get<0>(L);
  real const l10 = Get<2>(L);
  real const l11 = Get<3>(L);
  real const invL00 = 1_r / l00;
  real const invL11 = 1_r / l11;
  VMatrix2x2r const Linv{invL00, 0_r, -l10 * invL00 * invL11, invL11};
  VMatrix2x2r const B = Dot2x2(Linv, Dot2x2(S, Transpose2x2(Linv)));

  // Project B to be PSD using standard eigenvalue clamping.
  // After the congruence transformation, B's eigenvalues are the generalized eigenvalues
  // of S w.r.t. M, which have units of [S]/[M].
  VMatrix2x2r Bclamped = B;
  ProjectSymPsd(Bclamped, epsEigenvalue);

  // Transform back: S' = L * B' * L^T
  return Dot2x2(L, Dot2x2(Bclamped, Transpose2x2(L)));
}

/**************************************************************************************************
  Batched Eigendecomposition
*/

template <int kBatchSize>
inline void BatchedAnalyticalEigendecompSym3x3(
    BatchSymMatrix3x3<kBatchSize> const& sym,
    BatchReal3<kBatchSize>& eigvalues,
    BatchReal3x3<kBatchSize>* eigvecs) {
  BatchSymMatrix3x3<kBatchSize> normalizedSym = sym;
  auto const scale = detail::NormalizeSym(normalizedSym);
  if (eigvecs != nullptr) {
    detail::EigendecompSym3x3(normalizedSym, eigvalues, *eigvecs);
  } else {
    eigvalues = detail::EigenvalsSym3x3(normalizedSym);
  }
  eigvalues *= scale;
}

/**************************************************************************************************
  Batched SVD

  WARNING: Results may be inaccurate if 𝐅ᵀ𝐅 or det(𝐅) overflows or underflows.
*/

namespace detail {

// Compute G = F^T * F (symmetric, 6 unique entries).
template <int kBatchSize>
[[nodiscard]] MOCHI_FORCE_INLINE BatchSymMatrix3x3<kBatchSize> BatchedComputeGsym(
    BatchReal3x3<kBatchSize> const& fm) {
  return {
      fm[0][0] * fm[0][0] + fm[1][0] * fm[1][0] + fm[2][0] * fm[2][0],
      fm[0][1] * fm[0][1] + fm[1][1] * fm[1][1] + fm[2][1] * fm[2][1],
      fm[0][2] * fm[0][2] + fm[1][2] * fm[1][2] + fm[2][2] * fm[2][2],
      fm[0][0] * fm[0][1] + fm[1][0] * fm[1][1] + fm[2][0] * fm[2][1],
      fm[0][0] * fm[0][2] + fm[1][0] * fm[1][2] + fm[2][0] * fm[2][2],
      fm[0][1] * fm[0][2] + fm[1][1] * fm[1][2] + fm[2][1] * fm[2][2]};
}

// Compute U from F, VT (eigenvectors of G) and σ₀. Only VT rows 0 and 1 are used since U₂ =
// cross(U₀, U₁).
template <int kBatchSize>
MOCHI_FORCE_INLINE void BatchedComputeU(
    BatchReal3x3<kBatchSize> const& fm,
    BatchReal3x3<kBatchSize> const& VT,
    BatchReal<kBatchSize> const& sigma0,
    BatchReal3x3<kBatchSize>& U) {
  using V = BatchReal<kBatchSize>;
  using V3 = BatchReal3<kBatchSize>;
  using V3x3 = BatchReal3x3<kBatchSize>;
  real constexpr kEpsilon = std::numeric_limits<real>::epsilon();
  V const epsSqr = kEpsilon * kEpsilon;
  V const min = std::numeric_limits<real>::min();
  V const zero = V{0_r};
  V const one = V{1_r};
  V const invSigma0 = one / (sigma0 + min);

  // UT row 0: u0 = F * v0 / σ₀, unit to rounding, since σ₀² is the eigenvalue of G for v0.
  V3 u0 = DotMatVec(fm, VT[0]) * invSigma0;
  V const u0zero = sigma0 * sigma0 <= min;
  u0[0] = Select(u0zero, one, u0[0]);
  u0[1] = Select(u0zero, zero, u0[1]);
  u0[2] = Select(u0zero, zero, u0[2]);

  // UT row 1: u1 = F * v1 / σ₀ minus its component along u0. Rounding leaves that component at
  // about ε, which would tilt u1 by about εσ₀/σ₁, and which the rank test below would mistake for
  // a nonzero σ₁ if F is rank-deficient. Divided by σ₀, the test is relative, and a u1 that passes
  // it has a squared norm of at least ε², far from underflow.
  V3 u1 = DotMatVec(fm, VT[1]) * invSigma0;
  u1 -= Dot(u0, u1) * u0;

  V const u1normSqr = NormSqr(u1);
  V const u1invNorm = one / (Sqrt(u1normSqr) + min);
  V const u1zero = u1normSqr <= epsSqr;
  u1[0] *= u1invNorm;
  u1[1] *= u1invNorm;
  u1[2] *= u1invNorm;

  if (AnyTrue<kBatchSize>(u1zero)) MOCHI_UNLIKELY {
    // Fallback: OrthogonalVector(U[0])
    V3 const fo = Normalize(OrthogonalVector(u0));
    u1[0] = Select(u1zero, fo[0], u1[0]);
    u1[1] = Select(u1zero, fo[1], u1[1]);
    u1[2] = Select(u1zero, fo[2], u1[2]);
  }

  // UT row 2 = cross(UT[0], UT[1])
  V3 const u2 = Cross(u0, u1);

  // Transpose UT -> U.
  U = Transpose(V3x3{u0, u1, u2});
}

// Singular values from the eigenvalues of the normalized 𝐆 = 𝐅ᵀ𝐅 and its scale.
template <int kBatchSize>
[[nodiscard]] MOCHI_FORCE_INLINE BatchReal3<kBatchSize> BatchedSingularValues(
    BatchReal3x3<kBatchSize> const& F,
    BatchReal3<kBatchSize> const& normalizedEigvals,
    BatchReal<kBatchSize> const& scale) {
  using V = BatchReal<kBatchSize>;
  // σᵢ² = λᵢ. Clamp to ℝ⁺ for numerical safety.
  V const zero = {};
  BatchReal3<kBatchSize> Sg = {
      Sqrt(Max(normalizedEigvals[0] * scale, zero)),
      Sqrt(Max(normalizedEigvals[1] * scale, zero)),
      Sqrt(Max(normalizedEigvals[2] * scale, zero))};

  // If det(𝐅) < 0, encode inversion by negating the smallest singular value.
  Sg[2] = Select(Det(F) < zero, -Sg[2], Sg[2]);
  return Sg;
}

} // namespace detail

template <int kBatchSize>
inline void BatchedRotationVariantSvdVals3x3(
    BatchReal3x3<kBatchSize> const& F,
    BatchReal3<kBatchSize>& Sg) {
  // Form normal matrix 𝐆 = 𝐅ᵀ𝐅, normalize it to improve conditioning, then eigendecompose.
  BatchSymMatrix3x3<kBatchSize> Gsym = detail::BatchedComputeGsym<kBatchSize>(F);
  auto const scale = detail::NormalizeSym(Gsym);
  Sg = detail::BatchedSingularValues<kBatchSize>(F, detail::EigenvalsSym3x3(Gsym), scale);
}

template <int kBatchSize>
inline void BatchedRotationVariantSvdValsAndVT3x3(
    BatchReal3x3<kBatchSize> const& F,
    BatchReal3<kBatchSize>& Sg,
    BatchReal3x3<kBatchSize>& VT) {
  BatchSymMatrix3x3<kBatchSize> Gsym = detail::BatchedComputeGsym<kBatchSize>(F);
  auto const scale = detail::NormalizeSym(Gsym);
  BatchReal3<kBatchSize> eigvals MOCHI_NO_INIT;
  detail::EigendecompSym3x3(Gsym, eigvals, VT);
  Sg = detail::BatchedSingularValues<kBatchSize>(F, eigvals, scale);
}

template <int kBatchSize>
inline void BatchedRotationVariantSvd3x3(
    BatchReal3x3<kBatchSize> const& F,
    BatchReal3x3<kBatchSize>& U,
    BatchReal3<kBatchSize>& Sg,
    BatchReal3x3<kBatchSize>& VT) {
  BatchedRotationVariantSvdValsAndVT3x3<kBatchSize>(F, Sg, VT);
  detail::BatchedComputeU<kBatchSize>(F, VT, Sg[0], U);
}

template <int kBatchSize>
inline void BatchedProjectSymPsd(BatchReal3x3<kBatchSize>& A, real eps) {
  MOCHI_ASSERT_VERBOSE(eps >= 0_r, "Negative eps.");
  using V = BatchReal<kBatchSize>;
  using V3 = BatchReal3<kBatchSize>;
  using V3x3 = BatchReal3x3<kBatchSize>;
  using VSym3x3 = BatchSymMatrix3x3<kBatchSize>;

  // Extract symmetric part: sym = {A00, A11, A22, A01, A02, A12}.
  VSym3x3 const sym = {
      A[0][0],
      A[1][1],
      A[2][2],
      V{0.5_r} * (A[0][1] + A[1][0]),
      V{0.5_r} * (A[0][2] + A[2][0]),
      V{0.5_r} * (A[1][2] + A[2][1])};

  // Skip the eigendecomposition when every lane is already SPD based on strict Sylvester check on
  // the three leading principal minors.
  V const minor1 = sym[0];
  V const minor2 = sym[0] * sym[1] - sym[3] * sym[3];
  V const minor3 = sym[0] * (sym[1] * sym[2] - sym[5] * sym[5]) -
      sym[3] * (sym[3] * sym[2] - sym[5] * sym[4]) + sym[4] * (sym[3] * sym[5] - sym[1] * sym[4]);
  if (AllTrue<kBatchSize>(Min(minor1, minor2, minor3) > V{0_r})) {
    return;
  }

  // Eigendecomposition: QT stores eigenvectors as rows.
  V3 eigvals MOCHI_NO_INIT;
  V3x3 QT MOCHI_NO_INIT;
  BatchedAnalyticalEigendecompSym3x3<kBatchSize>(sym, eigvals, &QT);

  // Clamp eigenvalues to eps.
  V const vEps = eps;
  eigvals[0] = Max(eigvals[0], vEps);
  eigvals[1] = Max(eigvals[1], vEps);
  eigvals[2] = Max(eigvals[2], vEps);

  // Reconstruct: A = Q * diag(eigvals) * Q^T = (Q^T)^T * diag(eigvals) * Q^T.
  // D * Q^T: scale row i of Q^T by eigvals[i].
  V3x3 const DQT = {eigvals[0] * QT[0], eigvals[1] * QT[1], eigvals[2] * QT[2]};

  // A = Q * (D * Q^T) = transpose(Q^T) * (D * Q^T).
  A = Dot(Transpose(QT), DQT);
}

template <int kBatchSize>
[[nodiscard]] BatchReal2x2<kBatchSize> BatchedProjectPsdWithMetric(
    BatchReal2x2<kBatchSize> const& S,
    BatchReal2x2<kBatchSize> const& M,
    real epsCholesky,
    real epsEigenvalue) {
  using V = BatchReal<kBatchSize>;
  using V2x2 = BatchReal2x2<kBatchSize>;

  // Cholesky of M: M = L L^T where L is lower-triangular.
  V const vEpsChol = V{epsCholesky};
  V const l00 = Sqrt(Max(M[0][0], vEpsChol));
  V const invL00 = V{1_r} / l00;
  V const l10 = M[1][0] * invL00;
  V const l11 = Sqrt(Max(M[1][1] - l10 * l10, vEpsChol));
  V const invL11 = V{1_r} / l11;

  // L^{-1}
  V2x2 const Linv = {NdArray<V, 2>{invL00, V{0_r}}, NdArray<V, 2>{-l10 * invL00 * invL11, invL11}};

  // B = L^{-1} S L^{-T}
  V2x2 const B = Dot(Linv, Dot(S, Transpose(Linv)));

  // Compute eigenvalues of 2x2 symmetric B.
  V const halfTrace = 0.5_r * Trace(B);
  V const diff = B[0][0] - B[1][1];
  V const disc = Sqrt(Max(V{0.25_r} * diff * diff + B[0][1] * B[0][1], V{0_r}));
  V const eig0 = halfTrace + disc;
  V const eig1 = halfTrace - disc;

  // Select the row formulation with best numerical conditioning (avoids catastrophic cancellation).
  auto const b00GtB11 = (B[0][0] > B[1][1]);
  V const vxRaw = Select(b00GtB11, eig0 - B[1][1], B[0][1]);
  V const vyRaw = Select(b00GtB11, B[0][1], eig0 - B[0][0]);
  V const normRawSqr = Sqr(vxRaw) + Sqr(vyRaw);

  // The only remaining degenerate case is when the matrix is a perfect multiple of identity.
  V const vEps = std::numeric_limits<real>::epsilon();
  auto const isIsotropic = (normRawSqr <= Sqr(vEps) * Max(Sqr(B[0][0]), Sqr(B[1][1])));
  V const vx = Select(isIsotropic, V{1_r}, vxRaw);
  V const vy = Select(isIsotropic, V{0_r}, vyRaw);

  V const vMin = std::numeric_limits<real>::min();
  V const invNorm = V{1_r} / Max(Sqrt(vx * vx + vy * vy), vMin);
  V const nx = vx * invNorm;
  V const ny = vy * invNorm;

  // Eigenvalue clamping. Bclamped = B + d0 * n⊗n + d1 * m⊗m where m = (-ny, nx).
  V const vEpsEig = V{epsEigenvalue};
  V const d0 = Max(eig0, vEpsEig) - eig0;
  V const d1 = Max(eig1, vEpsEig) - eig1;

  V2x2 Bclamped MOCHI_NO_INIT;
  Bclamped[0][0] = B[0][0] + d0 * nx * nx + d1 * ny * ny;
  Bclamped[0][1] = B[0][1] + d0 * nx * ny - d1 * ny * nx;
  Bclamped[1][0] = Bclamped[0][1];
  Bclamped[1][1] = B[1][1] + d0 * ny * ny + d1 * nx * nx;

  // S' = L Bclamped L^T
  V2x2 const L = {NdArray<V, 2>{l00, V{0_r}}, NdArray<V, 2>{l10, l11}};
  return Dot(L, Dot(Bclamped, Transpose(L)));
}

} // namespace mochi
