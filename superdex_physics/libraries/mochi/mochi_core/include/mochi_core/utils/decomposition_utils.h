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

#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/batch_types.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/nd_array_utils.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/vmatrix.h>

#include <limits>

namespace mochi {

/**************************************************************************************************
  Eigendecomposition of Symmetric Matrices

  Performs the analytical eigendecomposition 𝐐𝚲𝐐ᵀ of a symmetric matrix 𝐀. As the matrix is
  symmetric, only the upper-right entries (◹) are used for the decomposition. Eigenvectors are
  normalized and stored as columns of 𝐐. If transpose is set to true, 𝐐ᵀ will be returned instead of
  𝐐.
*/

inline void AnalyticalEigendecompSym(
    Matrix2x2r mat,
    Real2& eigvalues,
    Matrix2x2r* eigvecs = nullptr,
    bool transpose = false);

inline void AnalyticalEigendecompSym(
    Matrix3x3r mat,
    Real3& eigvalues,
    Matrix3x3r* eigvecs = nullptr,
    bool transpose = false);

inline void
AnalyticalEigendecompSym2x2(VSymMatrix2x2r mat, Vec4r& eigvalues, VMatrix2x2r* eigvecs = nullptr);

inline void
AnalyticalEigendecompSym3x3(VSymMatrix3x3r mat, Vec4r& eigvalues, VMatrix3x3r* eigvecs = nullptr);

/**************************************************************************************************
  Singular-Value Decomposition

  Performs the rotation-variant singular value decomposition 𝐔𝚺𝐕ᵀ of a matrix 𝐅. Inversions i.e.
  when det(𝐅) < 0) are encoded as negative singular values on the smallest entry of 𝚺. 𝐔 and 𝐕 are
  guaranteed to be reflection-free rotations (i.e. det(𝐔) = det(𝐕) = 1). For the sake of efficiency,
  the function outputs 𝐕ᵀ instead of 𝐕.

  WARNING: May be inaccurate if the matrix entries are several orders of magnitude above or below 1.
*/

inline void RotationVariantSvdVals(Matrix3x3r const& F, Real3& Sg, Int3& order);

inline void RotationVariantSvdVecs(
    Matrix3x3r const& F,
    Int3 const& order,
    Real3& Sg,
    Matrix3x3r& U,
    Matrix3x3r& VT);

inline void RotationVariantSvd(Matrix3x3r const& F, Matrix3x3r& U, Real3& Sg, Matrix3x3r& VT);

inline void RotationVariantSvdVals3x3(VMatrix3x3r const& F, Vec4r& Sg, Int3& order);

inline void RotationVariantSvdVecs3x3(
    VMatrix3x3r const& F,
    Int3 const& order,
    Vec4r const& Sg,
    VMatrix3x3r& U,
    VMatrix3x3r& VT);

inline void RotationVariantSvd3x3(
    NdArray<Vec4r, 3> const& F,
    NdArray<Vec4r, 3>& U,
    Vec4r& Sg,
    NdArray<Vec4r, 3>& VT);

/**************************************************************************************************
  Polar Decomposition

  WARNING: May be inaccurate if the matrix entries are several orders of magnitude above or below 1.
*/

// Left polar decomposition of A = U * P, where U is orthogonal and P is symmetric positive
// semi-definite.
inline void LeftPolarDecomposition3x3(VMatrix3x3r const& A, VMatrix3x3r& U, VMatrix3x3r& P);

/**************************************************************************************************
  Positive Semi-Definite Projection of Symmetric Matrices.
*/

// Compute the positive semi-definite projection of a 2x2 symmetric matrix, in place.
// Note that in reality we don't set the minimum eigenvalue to zero but add a small perturbation.
inline void ProjectSymPsd(VMatrix2x2r& A, real eps = std::numeric_limits<real>::epsilon());

// Compute the positive semi-definite projection of a 3x3 symmetric matrix, in place.
// Note that in reality we don't set the minimum eigenvalue to zero but add a small perturbation.
inline void ProjectSymPsd(VMatrix3x3r& A, real eps = std::numeric_limits<real>::epsilon());

/**************************************************************************************************
  Cholesky Decomposition
*/

// Compute the lower-triangular Cholesky factor L of a symmetric 2x2 positive-definite matrix A,
// such that A = L * L^T. Returns L in full row-major representation (with upper-right entry zero).
// Uses epsilon clamping for numerical robustness when the matrix is nearly singular or not PD.
// Epsilon must be strictly positive to avoid divide-by-zero.
[[nodiscard]] inline VMatrix2x2r CholeskySym2x2(
    VMatrix2x2r const& A,
    real eps = std::numeric_limits<real>::epsilon());

/**************************************************************************************************
  Metric-Aware PSD Projection
*/
// Projects S to be positive semi-definite with respect to the inner product defined by M.
// The standard eigenvalue problem S*v = λ*v is not coordinate-invariant. Instead, we solve
// the generalized eigenvalue problem:
//   S * v = λ * M * v
//
// This is reduced to a standard eigenvalue problem via Cholesky factorization:
//   M = L * L^T
//   L^(-1) * S * L^(-T) * w = λ * w, where w = L^T * v
//
// The eigenvalues λ are then clamped to be non-negative, and S is reconstructed.
//
// @param[in] S The symmetric tensor to project.
// @param[in] M The metric tensor (must be symmetric positive definite).
// @param[in] epsCholesky Tolerance for Cholesky decomposition of M. This ensures numerical
//   stability when M is near-singular. Should scale with ||M||.
//   Typical value: `kRelTol * Norm(M)` where `kRelTol ~ 100 * machine_epsilon`.
// @param[in] epsEigenvalue Tolerance for clamping generalized eigenvalues. The generalized
//   eigenvalues λ satisfy S*v = λ*M*v, so λ has units of [S]/[M]. After the congruence
//   transformation, eigenvalues are O(||S||/||M||).
//   - If S and M have similar scales: use a small absolute value (e.g., `machine_epsilon`).
//   - If S and M have different scales: scale as `kRelTol * ||S|| / ||M||`.
//   Typical value for similar scales: `std::numeric_limits<real>::epsilon()`.
//
// @return The projected tensor S', which is PSD with respect to the metric M.
[[nodiscard]] inline VMatrix2x2r ProjectPsdWithMetric(
    VMatrix2x2r const& S,
    VMatrix2x2r const& M,
    real epsCholesky = std::numeric_limits<real>::epsilon(),
    real epsEigenvalue = std::numeric_limits<real>::epsilon());

/**************************************************************************************************
  Batched Decompositions

  Follow the same conventions as the non-batched overloads.
*/

template <int kBatchSize>
inline void BatchedAnalyticalEigendecompSym3x3(
    BatchReal6<kBatchSize> sym,
    BatchReal3<kBatchSize>& eigvalues,
    BatchReal3x3<kBatchSize>* eigvecs);

/// @brief Cached normalized eigensystem of F^T * F for split batched rotation-variant SVD.
///
/// Treat this as an opaque cache. It is only valid for the same F used to populate it with @ref
/// BatchedRotationVariantSvdVals3x3.
///
/// @see BatchedRotationVariantSvdVals3x3
template <int kBatchSize>
struct BatchedRotationVariantSvdNormalEigensystem3x3 {
  BatchReal6<kBatchSize> normalizedGsym;
  BatchReal3<kBatchSize> normalizedEigvals;
};

/// @brief Computes the rotation-variant singular values Sg of F, ordered by descending magnitude
/// (|Sg[0]| >= |Sg[1]| >= |Sg[2]|). The smallest entry Sg[2] is negated iff det(F) < 0.
template <int kBatchSize>
inline void BatchedRotationVariantSvdVals3x3(
    BatchReal3x3<kBatchSize> const& F,
    BatchReal3<kBatchSize>& Sg);

/// @brief Computes the rotation-variant singular values Sg of F and stores the normalized normal
/// eigensystem needed by @ref BatchedRotationVariantSvdVecs3x3.
///
/// This overload normalizes F^T * F before eigendecomposition and rescales Sg back to F units.
/// The singular values follow the same ordering and sign convention as the 2-argument overload.
/// Pass normalEigensystem unchanged to @ref BatchedRotationVariantSvdVecs3x3 with the same F to
/// compute the corresponding singular vectors.
template <int kBatchSize>
inline void BatchedRotationVariantSvdVals3x3(
    BatchReal3x3<kBatchSize> const& F,
    BatchReal3<kBatchSize>& Sg,
    BatchedRotationVariantSvdNormalEigensystem3x3<kBatchSize>& normalEigensystem);

/// @brief Computes rotation-variant singular vectors from a cached normalized normal eigensystem.
///
/// @warning normalEigensystem must be produced by the matching @ref
/// BatchedRotationVariantSvdVals3x3 call for the same F.
template <int kBatchSize>
inline void BatchedRotationVariantSvdVecs3x3(
    BatchReal3x3<kBatchSize> const& F,
    BatchedRotationVariantSvdNormalEigensystem3x3<kBatchSize> const& normalEigensystem,
    BatchReal3x3<kBatchSize>& U,
    BatchReal3x3<kBatchSize>& VT);

template <int kBatchSize>
inline void BatchedRotationVariantSvdValsAndVT3x3(
    BatchReal3x3<kBatchSize> const& F,
    BatchReal3<kBatchSize>& Sg,
    BatchReal3x3<kBatchSize>& VT);

template <int kBatchSize>
inline void BatchedRotationVariantSvd3x3(
    BatchReal3x3<kBatchSize> const& F,
    BatchReal3x3<kBatchSize>& U,
    BatchReal3<kBatchSize>& Sg,
    BatchReal3x3<kBatchSize>& VT);

/// @brief Batched PSD projection of a symmetric 3x3 matrix, in place.
template <int kBatchSize>
inline void BatchedProjectSymPsd(
    BatchReal3x3<kBatchSize>& A,
    real eps = std::numeric_limits<real>::epsilon());

/// @brief Batched PSD projection of a symmetric 2x2 tensor with respect to a metric.
template <int kBatchSize>
[[nodiscard]] inline BatchReal2x2<kBatchSize> BatchedProjectPsdWithMetric(
    BatchReal2x2<kBatchSize> const& S,
    BatchReal2x2<kBatchSize> const& M,
    real epsCholesky = std::numeric_limits<real>::epsilon(),
    real epsEigenvalue = std::numeric_limits<real>::epsilon());

} // namespace mochi

#include "decomposition_utils_inl.h"
