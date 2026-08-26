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
namespace detail {
/**************************************************************************************************
  Helper functions (Scalar)
*/

MOCHI_FORCE_INLINE constexpr void SortEigenValues2x2(Real2& eigvalues, Int2& order) {
  // Sort eigenvalues in descending order.
  // λ₂ > λ₁
  order = {0, 1};
  if (eigvalues[1] > eigvalues[0]) {
    std::swap(eigvalues[0], eigvalues[1]);
    std::swap(order[0], order[1]);
  }
}

MOCHI_FORCE_INLINE constexpr void SortEigenVectors2x2(Int2 const& order, Matrix2x2r& eigvecs) {
  Matrix2x2r const unsortedEigvecs = eigvecs;
  eigvecs[0] = unsortedEigvecs[order[0]];
  eigvecs[1] = unsortedEigvecs[order[1]];
}

MOCHI_FORCE_INLINE constexpr void SortEigenValues3x3(Real3& eigvalues, Int3& order) {
  // Sort eigenvalues in descending order.
  // λ₂ > λ₁
  order = {0, 1, 2};
  if (eigvalues[1] > eigvalues[0]) {
    std::swap(eigvalues[0], eigvalues[1]);
    std::swap(order[0], order[1]);
  }
  // λ₃ > λ₁
  if (eigvalues[2] > eigvalues[0]) {
    std::swap(eigvalues[0], eigvalues[2]);
    std::swap(order[0], order[2]);
  }
  // λ₃ > λ₂
  if (eigvalues[2] > eigvalues[1]) {
    std::swap(eigvalues[1], eigvalues[2]);
    std::swap(order[1], order[2]);
  }
}

MOCHI_FORCE_INLINE constexpr void SortEigenVectors3x3(Int3 const& order, Matrix3x3r& eigvecs) {
  Matrix3x3r const unsortedEigvecs = eigvecs;
  eigvecs[0] = unsortedEigvecs[order[0]];
  eigvecs[1] = unsortedEigvecs[order[1]];
  eigvecs[2] = unsortedEigvecs[order[2]];
}

/**************************************************************************************************
  Helper functions (SIMD)
*/

MOCHI_FORCE_INLINE void SortEigenValues2x2(Vec4r& eigvalues, Int2& order) {
  // Sort eigenvalues in descending order.
  // λ₂ > λ₁
  Vec4r shuffledEigvalues = Shuffle<1, 0, 2, 3>(eigvalues);
  if (AllTrue<1>(shuffledEigvalues >= eigvalues)) {
    eigvalues = shuffledEigvalues;
    order = {1, 0};
  } else {
    order = {0, 1};
  }
}

MOCHI_FORCE_INLINE void SortEigenVectors2x2(Int2 const& order, Vec4r& eigvecs) {
  if (order[0] == 1) {
    eigvecs = Shuffle<2, 3, 0, 1>(eigvecs);
  }
}

MOCHI_FORCE_INLINE void SortEigenValues3x3(Vec4r& eigvalues, Int3& order) {
  // Use the scalar implementation and let the compiler optimize it. This is faster than using an
  // explicit SIMD implementation, both on x86 and ARM.
  Real3 eigsAsReal3{eigvalues[0], eigvalues[1], eigvalues[2]};
  SortEigenValues3x3(eigsAsReal3, order);
  eigvalues = ToSimd(eigsAsReal3, 0_r);
}

MOCHI_FORCE_INLINE void SortEigenVectors3x3(Int3 const& order, VMatrix3x3r& eigvecs) {
  VMatrix3x3r unsortedEigvecs = eigvecs;
  eigvecs[0] = unsortedEigvecs[order[0]];
  eigvecs[1] = unsortedEigvecs[order[1]];
  eigvecs[2] = unsortedEigvecs[order[2]];
}

/**************************************************************************************************
  Eigendecomposition of Symmetric Matrices: 2x2 scalar
*/

// WARNING: Results may be inaccurate if the matrix entries are several orders of magnitude above or
// below 1.
MOCHI_FORCE_INLINE void
AnalyticalEigenvalsSym(Matrix2x2r const& mat, Real2& eigvalues, Int2& order) {
  // REFERENCE: Closed-form expressions of the eigen decomposition of 2 x 2 and 3 x 3 Hermitian
  // matrices. Deledalle, C. A., Denis, L., Tabti, S., & Tupin, F. (2017).
  constexpr auto kEpsilon = std::numeric_limits<real>::epsilon();

  // Fetch the symmetric terms.
  // | a c |
  // | c b |
  real const& a = mat[0][0];
  real const& c = mat[0][1];
  real const& b = mat[1][1];

  // Edge case: c≈0, therefore the matrix is already in diagonal form.
  if (NearEqual(c, 0.0_r, kEpsilon)) {
    eigvalues = {a, b};
  }

  // Standard case.
  else {
    real const delta = std::sqrt(4.0_r * c * c + (a - b) * (a - b));
    eigvalues[0] = (a + b - delta) * 0.5_r;
    eigvalues[1] = (a + b + delta) * 0.5_r;
  }

  // Sort the eigenvalues in descending order.
  SortEigenValues2x2(eigvalues, order);
}

// WARNING: Results may be inaccurate if the matrix entries are several orders of magnitude above or
// below 1.
MOCHI_FORCE_INLINE void AnalyticalEigenvecsSym(
    Matrix2x2r const& mat,
    Int2 const& order,
    Real2& eigvalues,
    Matrix2x2r& eigvecs,
    bool transpose) {
  // REFERENCE: Closed-form expressions of the eigen decomposition of 2 x 2 and 3 x 3
  // Hermitian matrices. Deledalle, C. A., Denis, L., Tabti, S., & Tupin, F. (2017).
  constexpr auto kEpsilon = std::numeric_limits<real>::epsilon();

  // Fetch the symmetric terms.
  // | a c |
  // | c b |
  real const& c = mat[0][1];
  real const& b = mat[1][1];

  // Edge case: c≈0, therefore the matrix is already in diagonal form.
  if (NearEqual(c, 0.0_r, kEpsilon)) {
    eigvecs = Eye<2>();
    SortEigenVectors2x2(order, eigvecs);

    // NOTE: For 2x2, every sorted-Eye permutation is symmetric. There is no need to transpose
    // eigvecs back if the transposed eigenvectors matrix wasn't requested.
  }

  // Regular case:
  else {
    // NOTE: This case does not require sorting as the eigenvectors are already extracted in the
    // correct order.
    // Store eigenvectors in rows (i.e. store 𝐐ᵀ).
    (eigvecs)[0] = Normalize(Real2{(eigvalues[0] - b) / c, 1.0_r});
    (eigvecs)[1] = Normalize(Real2{(eigvalues[1] - b) / c, 1.0_r});

    // If the transposed eigenvectors matrix wasn't requested,
    // transpose back so the eigenvectors are stored in columns (i.e. 𝐐)
    if (!transpose) {
      eigvecs = Transpose(eigvecs);
    }
  }
}

} // namespace detail

inline void
AnalyticalEigendecompSym(Matrix2x2r mat, Real2& eigvalues, Matrix2x2r* eigvecs, bool transpose) {
  real scale =
      Max((1_r / 4_r) * (Sum(Abs(mat[0])) + Sum(Abs(mat[1]))), std::numeric_limits<real>::min());
  mat /= scale; // Scale to improve numerical stability
  Int2 order;
  detail::AnalyticalEigenvalsSym(mat, eigvalues, order);
  if (eigvecs != nullptr) {
    detail::AnalyticalEigenvecsSym(mat, order, eigvalues, *eigvecs, transpose);
  }
  eigvalues *= scale;
}

/**************************************************************************************************
  Eigendecomposition of Symmetric Matrices: 3x3 scalar
*/

namespace detail {

// Compute ψ from x₁ and x₂ for the 3x3 symmetric eigenvalue formula.
// Reference: Deledalle et al. (2017), https://hal.science/hal-01501221/document
[[nodiscard]] MOCHI_FORCE_INLINE real ComputePsiSym3x3(real x1, real x2, real kEpsilon) {
  MOCHI_ASSERT_VERBOSE(x1 >= 0_r, "x1 must be non-negative.");
  real psi = kPI * 0.5_r;
  if (!NearZero(x2, kEpsilon)) {
    if (x1 > 1e10_r || Abs(x2) > 1e10_r)
      MOCHI_UNLIKELY {
        // NOTE @Jesus: This operation may overflow float for really messy
        // deformations. To solve this we transform the operation √(4x₁³−x₂²)/x₂
        // into √(4x₁³/x₂² - 1), and then group the factors of the first sumand
        // to keep values small x₁·(x₁·(2/x₂))²
        real k2divx2 = 2_r / x2;
        real x1mulk2divx2 = x1 * k2divx2; // (2·x1)/x2
        real x1mulk2divx2sqr = x1mulk2divx2 * x1mulk2divx2; // (4·x1·x1)/(x2·x2)
        MOCHI_ASSERT_VERBOSE(IsFinite(x1mulk2divx2sqr), "Float overflow");

        psi = x1 * x1mulk2divx2sqr - 1_r; // (4·x1·x1·x1)/(x2·x2) - 1
        MOCHI_ASSERT_VERBOSE(IsFinite(psi), "Float overflow");

        psi = Sqrt(Max(psi, 0_r)); // Conservative √(4x₁³/x₂² - 1)
        if (x2 < 0_r) {
          psi = -psi;
        }
      }
    else {
      // This path everything should be ok regardless of operations
      psi = Sqrt(Max(0_r, 4_r * (x1 * x1 * x1) - (x2 * x2))) / x2; // Regular √(4x₁³−x₂²)/x₂
    }

    psi = ATan(psi); // arctan(√(4x₁³−x₂²)/x₂)

    if (x2 < 0_r) {
      psi += kPI;
    }
  }
  return psi;
}

// WARNING: Results may be inaccurate if the matrix entries are several orders of magnitude above or
// below 1.
MOCHI_FORCE_INLINE void
AnalyticalEigenvalsSym(Matrix3x3r const& mat, Real3& eigvalues, Int3& order) {
  constexpr auto kEpsilon = std::numeric_limits<real>::epsilon();

  // Fetch the symmetric terms.
  // | a d f |
  // | d b e |
  // | f e c |
  real const& a = mat[0][0];
  real const& b = mat[1][1];
  real const& c = mat[2][2];
  real const& d = mat[0][1];
  real const& e = mat[1][2];
  real const& f = mat[0][2];

  // Edge case: |𝐀−diag(𝐀)|≈0, therefore matrix is already in diagonal form.
  if (NearEqual(d, 0.0_r, kEpsilon) && NearEqual(e, 0.0_r, kEpsilon) &&
      NearEqual(f, 0.0_r, kEpsilon)) {
    eigvalues = {a, b, c};
  }

  // Standard case:
  else {
    // REFERENCE: Closed-form expressions of the eigen decomposition of 2 x 2 and 3 x 3
    // Hermitian matrices. Deledalle, C. A., Denis, L., Tabti, S., & Tupin, F. (2017).
    // https://hal.science/hal-01501221/document
    real const aSqr = a * a;
    real const bSqr = b * b;
    real const cSqr = c * c;
    real const dSqr = d * d;
    real const eSqr = e * e;
    real const fSqr = f * f;
    real const ab = a * b;
    real const ac = a * c;
    real const bc = b * c;
    real const twoA = 2.0_r * a;
    real const twoB = 2.0_r * b;
    real const twoC = 2.0_r * c;

    real const x1 = Max(aSqr + bSqr + cSqr - ab - ac - bc + 3.0_r * (dSqr + fSqr + eSqr), 0.0_r);
    real const x2 = -(twoA - b - c) * (twoB - a - c) * (twoC - a - b) +
        9.0_r * ((twoC - a - b) * dSqr + (twoB - a - c) * fSqr + (twoA - b - c) * eSqr) -
        54.0_r * (d * e * f);
    MOCHI_ASSERT_VERBOSE(IsFinite(x1), "Float overflow");
    MOCHI_ASSERT_VERBOSE(IsFinite(x2), "Float overflow");

    real const psi = ComputePsiSym3x3(x1, x2, kEpsilon);

    real constexpr kPiOver3 = kPI / 3.0_r;
    real constexpr kOneOver3 = 1.0_r / 3.0_r;
    real const psiOver3 = psi * kOneOver3;
    real const twoSqrtX1 = 2.0_r * Sqrt(x1);
    eigvalues[0] = (a + b + c - twoSqrtX1 * Cos(psiOver3)) * kOneOver3;
    eigvalues[1] = (a + b + c + twoSqrtX1 * Cos(psiOver3 - kPiOver3)) * kOneOver3;
    eigvalues[2] = (a + b + c + twoSqrtX1 * Cos(psiOver3 + kPiOver3)) * kOneOver3;
  }

  // Sort the eigenvalues in descending order.
  SortEigenValues3x3(eigvalues, order);
}

// WARNING: Results may be inaccurate if the matrix entries are several orders of magnitude above or
// below 1.
MOCHI_FORCE_INLINE void AnalyticalEigenvecsSym(
    Matrix3x3r const& mat,
    Int3 const& order,
    Real3& eigvalues,
    Matrix3x3r& eigvecs,
    bool transpose) {
  constexpr auto kEpsilon = std::numeric_limits<real>::epsilon();

  // Fetch the symmetric terms.
  // | a d f |
  // | d b e |
  // | f e c |
  real const& d = mat[0][1];
  real const& e = mat[1][2];
  real const& f = mat[0][2];

  // Edge case: |𝐀−diag(𝐀)|≈0, therefore matrix is already in diagonal form.
  if (NearEqual(d, 0.0_r, kEpsilon) && NearEqual(e, 0.0_r, kEpsilon) &&
      NearEqual(f, 0.0_r, kEpsilon)) {
    eigvecs = Eye<3>();
    SortEigenVectors3x3(order, eigvecs);
    // Ensure eigenvectors don't encode a reflection.
    // TODO: Doing the full cross product seems dumb but it's not slower than eigvecs[2] *= -1_r
    // with branching. However, there must be a better way.
    eigvecs[2] = Cross(eigvecs[0], eigvecs[1]);
  }

  // Standard case:
  else {
    // NOTE: This case does not require sorting as the eigenvectors are already extracted in the
    // correct order.

    // Compute the eigenvectors.
    // Deledalle et al.'s approach fails when matrix is in tridiagonal form (i.e. e≈0).
    // Instead, follow the approach used in PhysBAM to compute the eigenvectors.
    //
    // Fetch references to the eigenvectors.
    Real3& eigvec1 = (eigvecs)[0];
    Real3& eigvec2 = (eigvecs)[1];
    Real3& eigvec3 = (eigvecs)[2];

    // Flip if necessary so that first eigenvalue is the most different.
    real lambdaFlipX = eigvalues[0];
    real lambdaFlipZ = eigvalues[2];
    bool const flipped = (eigvalues[0] - eigvalues[1]) < (eigvalues[1] - eigvalues[2]);
    if (flipped) {
      std::swap(lambdaFlipX, lambdaFlipZ);
    }

    // Compute first eigenvector.
    // This approach for computing the eigenvector is based on the neat trick described here:
    // https://ocw.mit.edu/ans7870/18/18.013a/textbook/HTML/chapter04/section06.html
    {
      // NOTE: As 𝐀ᵀ = 𝐀, we can work with rows instead of columns for better
      // memory access patterns.
      Matrix3x3r const charPolyL1 = mat - DiagonalMatrix<3>(lambdaFlipX);
      Matrix3x3r const cofMatMinusL1 = Cofactor(charPolyL1);

      real largestRowSqrNorm = 0_r;
      eigvec1 = LargestRow(cofMatMinusL1, &largestRowSqrNorm);

      // EDGE CASE: characteristic polynomial matrix is zero.
      // Impose arbitrary eigenvector.
      if (NearEqual(largestRowSqrNorm, 0.0_r, kEpsilon)) {
        eigvec1 = BasisVector<real, 3>(0); // (1, 0, 0)
      }

      // Otherwise, ensure that the eigenvector is normalized.
      else {
        eigvec1 = Normalize(eigvec1, largestRowSqrNorm);
      }
    }

    // Form basis for orthogonal complement to the first eigenvector, and reduce
    // mat to this space.
    {
      // NOTE: As 𝐀ᵀ = 𝐀, we can work with rows instead of columns for better
      // memory access patterns.

      // Build transposed orthonormal basis.
      Matrix2x3r orthoBasis;
      orthoBasis[0] = Normalize(OrthogonalVector(eigvec1));
      orthoBasis[1] = Cross(eigvec1, orthoBasis[0]);

      // Reduce matrix to subspace spanned by this orthonormal basis.
      Matrix2x2r const reducedMat = Dot(orthoBasis, Dot(mat, Transpose(orthoBasis)));
      Matrix2x2r const charPolyL3 = reducedMat - DiagonalMatrix<2>(lambdaFlipZ);
      Matrix2x2r const cofReducedMatMinusL3 = Cofactor(charPolyL3);

      // Find eigenvector in reduced (orthogonal) space.
      real largestRowSqrNorm = 0_r;
      Real2 reducedEigvec3 = LargestRow(cofReducedMatMinusL3, &largestRowSqrNorm);

      // EDGE CASE: characteristic polynomial matrix is zero.
      // Impose arbitrary eigenvector.
      if (NearEqual(largestRowSqrNorm, 0.0_r, kEpsilon)) {
        reducedEigvec3 = BasisVector<real, 2>(0);
      }

      // Otherwise, ensure that the eigenvector is normalized.
      else {
        reducedEigvec3 = Normalize(reducedEigvec3, largestRowSqrNorm);
      }

      // Bring back to full space.
      eigvec3 = DotVecMat(reducedEigvec3, orthoBasis);
    }

    MOCHI_ASSERT_VERBOSE(
        NearZero(Dot(eigvec1, eigvec3), kEpsilon), "These vectors should already be orthogonal.");

    // Finally, the second eigenvector has to be perpendicular to the first and third.
    // Compute it via the cross product.
    eigvec2 = Cross(eigvec3, eigvec1);

    // Ensure unit length
    MOCHI_ASSERT_VERBOSE(
        NearEqual(1_r, NormSqr(eigvec1)) && NearEqual(1_r, NormSqr(eigvec2)) &&
            NearEqual(1_r, NormSqr(eigvec3)),
        "Expected unit-length vectors");

    // Unflip eigenvectors in case we flipped them.
    if (flipped) {
      auto temp = eigvec3;
      eigvec3 = -eigvec1;
      eigvec1 = temp;
    }
  }

  // Eigenvectors are stored as rows (𝐐ᵀ). If 𝐐 was requested, transpose back.
  if (!transpose) {
    eigvecs = Transpose(eigvecs);
  }
}

} // namespace detail

inline void
AnalyticalEigendecompSym(Matrix3x3r mat, Real3& eigvalues, Matrix3x3r* eigvecs, bool transpose) {
  real scale = (1_r / 9_r) * (Sum(Abs(mat[0])) + Sum(Abs(mat[1])) + Sum(Abs(mat[2])));
  scale = Max(scale, std::numeric_limits<real>::min());
  mat /= scale; // Scale to improve numerical stability
  Int3 order;
  detail::AnalyticalEigenvalsSym(mat, eigvalues, order);
  if (eigvecs != nullptr) {
    detail::AnalyticalEigenvecsSym(mat, order, eigvalues, *eigvecs, transpose);
  }
  eigvalues *= scale;
}

/**************************************************************************************************
  Eigendecomposition of Symmetric Matrices: 2x2 SIMD
*/

namespace detail {

// WARNING: Results may be inaccurate if the matrix entries are several orders of magnitude above or
// below 1.
MOCHI_FORCE_INLINE void
AnalyticalEigenvalsSym2x2(VSymMatrix2x2r const& mat, Vec4r& eigvalues, Int2& order) {
  // REFERENCE: Closed-form expressions of the eigen decomposition of 2 x 2 and 3 x 3
  // Hermitian matrices. Deledalle, C. A., Denis, L., Tabti, S., & Tupin, F. (2017)

  // INPUTS:
  // We assume that mat contains the complete matrix with on and off-diagonal
  // terms, following the sequence (a, b, c, ?). The matrix is in the form:
  // | a c |
  // | c b |

  // OUTPUTS:
  // Eigvalues 𝚲 are stored in the xy components. zw components can have arbitrary values
  // and must be ignored. Eigvectors 𝐐 are stored in the xy and zw components respectively

  real constexpr kEpsilon = std::numeric_limits<real>::epsilon();

  // Edge case: c≈0, therefore the matrix is already in diagonal form
  if (NearZero(Get<1>(mat), kEpsilon)) {
    // Copy entries a and b
    eigvalues = Shuffle<0, 2, 1, 3>(mat);
  }

  // Standard case.
  else {
    // clang-format off

    // Compute delta.
    Vec4r halfMat = Shuffle<0, 2, 1, 3>(mat * 0.5_r);               // 1/2 (a, b, c, ?)
    Vec4r halfMatS = Shuffle<1, 0, 2, 3>(halfMat);                  // 1/2 (b, a, c, ?)
    Vec4r temp0 = halfMat + halfMatS;                               // 1/2 (a + b, a + b, 2c, ?)
    Vec4r temp1 = halfMat - halfMatS;                               // 1/2 (a - b, b - a, 0, 0)
    Vec4r temp2 = Broadcast<1>(mat);                                // (c, c, c, c)
    Vec4r delta = Sqrt(temp1 * temp1 + temp2 * temp2);              // (Δ, Δ, ?, ?) = √(c² + 1/4 (a − b)²)

    // Compute eigenvalues
    eigvalues = temp0 + Neg<true, false, false, false>(delta);      // (λ₁, λ₂, ?, ?)

    // clang-format on
  }

  // Sort the eigenvalues in descending order.
  SortEigenValues2x2(eigvalues, order);
}

// WARNING: Results may be inaccurate if the matrix entries are several orders of magnitude above or
// below 1.
MOCHI_FORCE_INLINE void AnalyticalEigenvecsSym2x2(
    VSymMatrix2x2r const& mat,
    Int2 const& order,
    Vec4r& eigvalues,
    VMatrix2x2r& eigvecs) {
  real constexpr kEpsilon = std::numeric_limits<real>::epsilon();

  // Edge case: c≈0, therefore the matrix is already in diagonal form.
  if (NearZero(Get<1>(mat), kEpsilon)) {
    eigvecs = Vec4r(1.0_r, 0.0_r, 0.0_r, 1.0_r);
    SortEigenVectors2x2(order, eigvecs);
  }

  // Standard case:
  else {
    // NOTE: This case does not require sorting as the eigenvectors are already extracted in the
    // correct order.
    // clang-format off
    Vec4r temp2 = Broadcast<1>(mat);                      // (c, c, c, c)

    // Compute eigenvectors.
    Vec4r temp3 = Broadcast<2>(mat);                      // (b, b, b, b)
    eigvecs = (eigvalues - temp3) / temp2;                // ((λ₁ − b) / c, (λ₂ − b) / c, ?, ?)
    eigvecs = Shuffle<0, 2, 1, 3>(eigvecs);               // ((λ₁ − b) / c, ?, (λ₂ − b) / c, ?)
    eigvecs = Blend<0, 1, 0, 1>(eigvecs, Vec4r{1_r});     // ((λ₁ − b) / c, 1, (λ₂ − b) / c, 1) = (q₀₀, q₀₁, q₁₀, q₁₁)

    Vec4r temp4 = eigvecs * eigvecs;                      // (q₀₀²,        q₀₁²,        q₁₀²,        q₁₁²)
    Vec4r temp5 = Shuffle<1, 0, 3, 2>(temp4);             // (q₀₁²,        q₀₀²,        q₁₁²,        q₁₀²)
    Vec4r temp6 = temp4 + temp5;                          // (q₀₀² + q₀₁², q₀₀² + q₀₁², q₁₀² + q₁₁², q₁₀² + q₁₁²)
    Vec4r vSmall = std::numeric_limits<real>::min();      // (ε, ε, ε, ε)
    eigvecs /= (Sqrt(temp6) + vSmall);                    // (q₀₀ / |Q₀|, q₀₁ / |Q₀|, q₁₀ / |Q₁|, q₁₁ / |Q₁|)
                                   // clang-format on
  }
}

} // namespace detail

inline void
AnalyticalEigendecompSym2x2(VSymMatrix2x2r mat, Vec4r& eigvalues, VMatrix2x2r* eigvecs) {
  Vec4r scale = Max((1_r / 4_r) * HSum(Abs(mat)), std::numeric_limits<real>::min());
  mat /= scale; // Scale to improve numerical stability
  Int2 order;
  detail::AnalyticalEigenvalsSym2x2(mat, eigvalues, order);
  if (eigvecs != nullptr) {
    detail::AnalyticalEigenvecsSym2x2(mat, order, eigvalues, *eigvecs);
  }
  eigvalues *= scale;
}

/**************************************************************************************************
  Eigendecomposition of Symmetric Matrices: 3x3 SIMD
*/

namespace detail {

// WARNING: Results may be inaccurate if the matrix entries are several orders of magnitude above or
// below 1.
MOCHI_FORCE_INLINE void
AnalyticalEigenvalsSym3x3(VSymMatrix3x3r const& mat, Vec4r& eigvalues, Int3& order) {
  real constexpr kEpsilon = std::numeric_limits<real>::epsilon();

  // Edge case: |𝐀−diag(𝐀)|≈0, therefore matrix is already in diagonal form.
  if (NearZero<3>(mat[1], kEpsilon)) {
    eigvalues = mat[0];
  }

  // Standard case:
  else {
    // REFERENCE: Closed-form expressions of the eigen decomposition of 2 x 2 and 3 x 3
    // Hermitian matrices. Deledalle, C. A., Denis, L., Tabti, S., & Tupin, F. (2017).
    // https://hal.science/hal-01501221/document

    // NOTE: This case does not require sorting as the eigenvectors are already extracted in the
    // correct order.
    // clang-format off

    // Symmetric matrix entries.
    Vec4r const& abc = mat[0];                                                  // (a, b, c, ?)
    Vec4r const& dfe = mat[1];                                                  // (d, f, e, ?)

    // Compute x₁ = a² + b² + c² − ab − ac − bc + 3 (d² + f² + e²)
    Vec4r abAcBc = Shuffle<0, 0, 1, 3>(abc) * Shuffle<1, 2, 2, 3>(abc);         // (ab, ac, bc, ?)
    Vec4r dfeSqr = dfe * dfe;                                                   // (d², f², e², ?)
    real x1 = HSum<3>(abc * abc + 3_r * dfeSqr - abAcBc);                       // x₁
    MOCHI_ASSERT_VERBOSE(IsFinite(x1), "Float overflow");
    x1 = Max(x1, 0_r);

    // Compute x₂ = 9 (d²(2c − a − b) + f²(2b − a − c) + e²(2a − b − c)) -
    //              54 (def) - (2a - b - c) (2b - a - c) (2c - b)
    Vec4r temp0 = (abc + abc) - Shuffle<1, 2, 0, 3>(abc);                       // (2a - b, 2b - c, 2c - a, ?)
    Vec4r temp1 = temp0 - Shuffle<2, 0, 1, 3>(abc);                             // (2a - b - c, 2b - a - c, 2c - a - b, ?)
    Vec4r temp2 = temp1 * Shuffle<2, 1, 0, 3>(dfeSqr);                          // (e²(2a − b − c), f²(2b − a − c), d²(2c − a − b), ?)
    real x2 = 9_r * HSum<3>(temp2) - 54_r * HProd<3>(dfe) - HProd<3>(temp1);    // x₂
    MOCHI_ASSERT_VERBOSE(IsFinite(x2), "Float overflow");

    // clang-format on

    // Compute ψ.
    real const psi = ComputePsiSym3x3(x1, x2, kEpsilon);

    // Compute eigenvalues 𝚲.
    Vec4r temp3 = Neg<true, false, false, false>(Vec4r{Sqrt(x1 * 4_r)}); // 2(-√x₁, √x₁, √x₁, √x₁)
    Vec4r temp4{0_r, kPI / -3_r, kPI / 3_r};
    Vec4r temp5 = (psi / 3_r) + temp4;
    eigvalues = (temp3 * Cos(temp5) + HSum<3>(abc)) / 3_r;
  }

  // Sort the eigenvalues in descending order.
  SortEigenValues3x3(eigvalues, order);
}

// WARNING: Results may be inaccurate if the matrix entries are several orders of magnitude above or
// below 1.
MOCHI_FORCE_INLINE void AnalyticalEigenvecsSym3x3(
    VSymMatrix3x3r const& mat,
    Int3 const& order,
    Vec4r const& eigvalues,
    VMatrix3x3r& eigvecs) {
  real constexpr kEpsilon = std::numeric_limits<real>::epsilon();

  // Edge case: |𝐀−diag(𝐀)|≈0, therefore matrix is already in diagonal form.
  if (NearZero<3>(mat[1], kEpsilon)) {
    eigvecs = VEye<3>();
    SortEigenVectors3x3(order, eigvecs);
    // Ensure eigenvectors don't encode a reflection.
    // TODO: Doing the full cross product seems dumb but it's not slower than eigvecs[2] *= -1_r
    // with branching. However, there must be a better way.
    eigvecs[2] = Cross3(eigvecs[0], eigvecs[1]);
  }

  // Standard case:
  else {
    // REFERENCE: Closed-form expressions of the eigen decomposition of 2 x 2 and 3 x 3
    // Hermitian matrices. Deledalle, C. A., Denis, L., Tabti, S., & Tupin, F. (2017).

    // NOTE: This case does not require sorting as the eigenvectors are already extracted in the
    // correct order.

    // Helper constants.
    Vec4r const vEpsilon = kEpsilon;

    // Compute the eigenvectors.
    // Deledalle et al.'s approach fails when matrix is in tridiagonal form (i.e. e≈0).
    // Instead, follow the approach used in PhysBAM to compute the eigenvectors.

    // Flip if necessary so that first eigenvalue is the most different.
    Vec4r eigvaluesDiff = eigvalues - Shuffle<1, 2, 3, 0>(eigvalues);
    bool const flipped = Get<0>(eigvaluesDiff) < Get<1>(eigvaluesDiff);
    Vec4r lambdaFlip = flipped ? Shuffle<2, 1, 0, 3>(eigvalues) : eigvalues;

    // Compute first eigenvector.
    // NOTE: As 𝐀ᵀ = 𝐀, we can work with rows instead of columns for better memory access
    // patterns.
    Vec4r diagMinusL1 = mat[0] - Broadcast<0>(lambdaFlip);
    VSymMatrix3x3r cofMatMinusL1 = CofactorSym3x3(VSymMatrix3x3r{diagMinusL1, mat[1]});

    Vec4r largestRowSqrNorm;
    Vec4r eigvec1 = LargestRowColSym3x3(cofMatMinusL1, largestRowSqrNorm);
    eigvec1 = Normalize(eigvec1, largestRowSqrNorm);

    // EDGE CASE: characteristic polynomial matrix is zero. Impose arbitrary eigenvector.
    eigvec1 = Select(largestRowSqrNorm <= vEpsilon, SimdBasisVector<0>(), eigvec1);

    // Form basis for orthogonal complement to the first eigenvector, and reduce mat to this space.
    // NOTE: As 𝐀ᵀ = 𝐀, we can work with rows instead of columns for better
    // memory access patterns.

    // Build tranposed orthonormal basis.
    VMatrix2x3r orthoBasis;
    orthoBasis[0] = Normalize<3>(OrthogonalVector3(eigvec1));
    orthoBasis[1] = Cross3(eigvec1, orthoBasis[0]);

    // Reduce matrix to subspace spanned by this orthonormal basis.
    //
    // This is computed as :
    // Matrix2x2r const reducedMat = Dot(orthoBasis, Dot(mat, Transpose(orthoBasis)));
    //
    // Corresponds to:
    // [ux*(a*ux + d*uy + e*uz) + uy*(b*uy + d*ux + f*uz) + uz*(c*uz + e*ux + f*uy),
    //  vx*(a*ux + d*uy + e*uz) + vy*(b*uy + d*ux + f*uz) + vz*(c*uz + e*ux + f*uy)],
    // [ux*(a*vx + d*vy + e*vz) + uy*(b*vy + d*vx + f*vz) + uz*(c*vz + e*vx + f*vy),
    //  vx*(a*vx + d*vy + e*vz) + vy*(b*vy + d*vx + f*vz) + vz*(c*vz + e*vx + f*vy)]
    //
    // With entries:
    // a00 = a*ux^2 + b*uy^2 + c*uz^2 + 2*d*ux*uy + 2*e*ux*uz + 2*f*uy*uz
    // a11 = a*vx^2 + b*vy^2 + c*vz^2 + 2*d*vx*vy + 2*e*vx*vz + 2*f*vy*vz
    // a01 = a*vx*ux + b*vy*uy + c*vz*uz + d*(ux*vy + uy*vx) + e*(uz*vx + ux*vz) + f*(uy*vz + uz*vy)
    //
    Vec4r ux2_uy2_uz2 = orthoBasis[0] * orthoBasis[0];
    Vec4r vx2_vy2_vz2 = orthoBasis[1] * orthoBasis[1];
    Vec4r uxvx_uyvy_uzvz = orthoBasis[0] * orthoBasis[1];
    Vec4r uxuy_uxuz_uyuz = Shuffle<0, 0, 1, 3>(orthoBasis[0]) * Shuffle<1, 2, 2, 3>(orthoBasis[0]);
    Vec4r vxvy_vxvz_vyvz = Shuffle<0, 0, 1, 3>(orthoBasis[1]) * Shuffle<1, 2, 2, 3>(orthoBasis[1]);
    Vec4r uxvy_uzvx_uyvz = Shuffle<0, 2, 1, 3>(orthoBasis[0]) * Shuffle<1, 0, 2, 3>(orthoBasis[1]);
    Vec4r uyvx_uxvz_uzvy = Shuffle<1, 0, 2, 3>(orthoBasis[0]) * Shuffle<0, 2, 1, 3>(orthoBasis[1]);

    Vec4r a00 = mat[1] * uxuy_uxuz_uyuz;
    a00 += a00 + mat[0] * ux2_uy2_uz2;
    a00 = HSum<3>(a00);

    Vec4r a11 = mat[1] * vxvy_vxvz_vyvz;
    a11 += a11 + mat[0] * vx2_vy2_vz2;
    a11 = HSum<3>(a11);

    Vec4r a01 = mat[1] * (uxvy_uzvx_uyvz + uyvx_uxvz_uzvy) + mat[0] * uxvx_uyvy_uzvz;
    a01 = HSum<3>(a01);

    VSymMatrix2x2r reducedMat = Blend<0, 1, 1, 0>(a00, Blend<0, 0, 1, 0>(a01, a11));

    VSymMatrix2x2r cofReducedMatMinusL3 = Blend<0, 1, 0, 1>(Broadcast<2>(lambdaFlip), SimdZero());
    cofReducedMatMinusL3 = reducedMat - cofReducedMatMinusL3;
    cofReducedMatMinusL3 = CofactorSym2x2(cofReducedMatMinusL3);

    // Find eigenvector in reduced (orthogonal) space.
    Vec4r eigvec3 = LargestRowColSym2x2(cofReducedMatMinusL3, largestRowSqrNorm);
    eigvec3 = Normalize(eigvec3, largestRowSqrNorm);

    // EDGE CASE: characteristic polynomial matrix is zero.
    // Impose arbitrary eigenvector.
    eigvec3 = Select(largestRowSqrNorm <= vEpsilon, SimdBasisVector<0>(), eigvec3);

    // Bring eigenvector back to full space.
    eigvec3 = DotVecMat2x3(eigvec3, orthoBasis);

    // Finally, the second eigenvector has to be perpendicular to the first and third.
    // Compute it via the cross product.
    Vec4r eigvec2 = Cross3(eigvec3, eigvec1);

    // Unflip eigenvectors in case we flipped them.
    if (flipped) {
      eigvecs = VMatrix3x3r{eigvec3, eigvec2, -eigvec1};
    } else {
      eigvecs = VMatrix3x3r{eigvec1, eigvec2, eigvec3};
    }
  }
}

} // namespace detail

inline void
AnalyticalEigendecompSym3x3(VSymMatrix3x3r mat, Vec4r& eigvalues, VMatrix3x3r* eigvecs) {
  Int3 order;
  real scale = (1_r / 6_r) * (HSum<3>(Abs(mat[0]) + Abs(mat[1])));
  scale = Max(scale, std::numeric_limits<real>::min());
  mat /= scale; // Scale to improve numerical stability
  detail::AnalyticalEigenvalsSym3x3(mat, eigvalues, order);
  if (eigvecs != nullptr) {
    detail::AnalyticalEigenvecsSym3x3(mat, order, eigvalues, *eigvecs);
  }
  eigvalues *= scale;
}

/**************************************************************************************************
  Singular-Value Decomposition: Scalar
*/

// WARNING: Results may be inaccurate if the matrix entries are several orders of magnitude above or
// below 1.
inline void RotationVariantSvdVals(Matrix3x3r const& F, Real3& Sg, Int3& order) {
  // Form normal matrix 𝐆 = 𝐅ᵀ𝐅. Perform eigendecomposition of 𝐆.
  Real3 eigvalues;
  Matrix3x3r G = Dot(Transpose(F), F);
  detail::AnalyticalEigenvalsSym(G, eigvalues, order);

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

// WARNING: Results may be inaccurate if the matrix entries are several orders of magnitude above or
// below 1.
inline void RotationVariantSvdVecs(
    Matrix3x3r const& F,
    Int3 const& order,
    Real3& Sg,
    Matrix3x3r& U,
    Matrix3x3r& VT) {
  Real3 eigvals = Sg * Sg;
  Matrix3x3r G = Dot(Transpose(F), F);
  detail::AnalyticalEigenvecsSym(G, order, eigvals, VT, true);

  // Compute singular vectors 𝐔=𝐅𝐕𝚺⁻¹

  // Compute first row of UT
  U[0] = DotMatVec(F, VT[0]);
  real u0norm = Norm(U[0]);
  if (u0norm < std::numeric_limits<real>::epsilon())
    MOCHI_UNLIKELY {
      U[0] = Real3{1.0_r, 0.0_r, 0.0_r};
    }
  else {
    U[0] /= u0norm;
  }

  // Compute second row of UT
  U[1] = DotMatVec(F, VT[1]);
  real u1norm = Norm(U[1]);
  if (u1norm < std::numeric_limits<real>::epsilon())
    MOCHI_UNLIKELY {
      U[1] = Normalize(OrthogonalVector(U[0]));
    }
  else {
    U[1] /= u1norm;
  }

  // Compute third row of UT
  U[2] = Cross(U[0], U[1]);

  // Transpose to return U
  U = Transpose(U);
}

// WARNING: Results may be inaccurate if the matrix entries are several orders of magnitude above or
// below 1.
inline void RotationVariantSvd(Matrix3x3r const& F, Matrix3x3r& U, Real3& Sg, Matrix3x3r& VT) {
  // Form normal matrix 𝐆 = 𝐅ᵀ𝐅. Perform eigendecomposition of 𝐆.
  Real3 eigvalues;
  Matrix3x3r G = Dot(Transpose(F), F);
  AnalyticalEigendecompSym(G, eigvalues, &VT, true);

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

  // Compute first row of UT
  U[0] = DotMatVec(F, VT[0]);
  real u0norm = Norm(U[0]);
  if (u0norm < std::numeric_limits<real>::epsilon())
    MOCHI_UNLIKELY {
      U[0] = Real3{1.0_r, 0.0_r, 0.0_r};
    }
  else {
    U[0] /= u0norm;
  }

  // Compute second row of UT
  U[1] = DotMatVec(F, VT[1]);
  real u1norm = Norm(U[1]);
  if (u1norm < std::numeric_limits<real>::epsilon())
    MOCHI_UNLIKELY {
      U[1] = Normalize(OrthogonalVector(U[0]));
    }
  else {
    U[1] /= u1norm;
  }

  // Compute third row of UT
  U[2] = Cross(U[0], U[1]);

  // Transpose to return U
  U = Transpose(U);
}

/**************************************************************************************************
  Singular-Value Decomposition: SIMD
*/

// WARNING: Results may be inaccurate if the matrix entries are several orders of magnitude above or
// below 1.
inline void RotationVariantSvdVals3x3(VMatrix3x3r const& F, Vec4r& Sg, Int3& order) {
  // Compute F^T*F
  VSymMatrix3x3r Gsym = SimdFullToSym(Dot3x3(Transpose3x3(F), F));

  Vec4r eigvalues;
  detail::AnalyticalEigenvalsSym3x3(Gsym, eigvalues, order);

  // Compute the singular values (i.e. sqrt(eigvalues), clamp
  // them before computing sqrt in case there is rounding error
  // making them negative
  Sg = Sqrt(Max(eigvalues, Vec4r{0_r}));

  // If an inversion happened (i.e. det(𝐅) < 0), encode this
  // information by negating the smallest singular value.
  if (Det3x3(F) < 0.0) {
    Sg = Neg<0, 0, 1, 0>(Sg); // Negate flagged entries
  }
}

// WARNING: Results may be inaccurate if the matrix entries are several orders of magnitude above or
// below 1.
inline void RotationVariantSvdVecs3x3(
    VMatrix3x3r const& F,
    Int3 const& order,
    Vec4r const& Sg,
    VMatrix3x3r& U,
    VMatrix3x3r& VT) {
  // Compute F^T * F
  VMatrix3x3r FT = Transpose3x3(F);
  VSymMatrix3x3r Gsym = SimdFullToSym(Dot3x3(FT, F));

  Vec4r eigvalues = Sg * Sg;
  detail::AnalyticalEigenvecsSym3x3(Gsym, order, eigvalues, VT);

  // Compute singular vectors 𝐔=𝐅𝐕𝚺⁻¹.
  real constexpr kEpsilon = std::numeric_limits<real>::epsilon();

  // Compute first row of UT
  U[0] = DotVecMat3x3(VT[0], FT); // Faster than DotMatVec3x3
  real const u0normSqr = NormSqr<3>(U[0]);
  if (u0normSqr > kEpsilon)
    MOCHI_LIKELY {
      U[0] = Normalize(U[0], u0normSqr);
    }
  else
    MOCHI_UNLIKELY {
      U[0] = SimdBasisVector<0>();
    }

  // Compute second row of UT
  U[1] = DotVecMat3x3(VT[1], FT); // Faster than DotMatVec3x3
  real const u1normSqr = NormSqr<3>(U[1]);
  if (u1normSqr > kEpsilon)
    MOCHI_LIKELY {
      U[1] = Normalize(U[1], u1normSqr);
    }
  else
    MOCHI_UNLIKELY {
      U[1] = Normalize<3>(OrthogonalVector3(U[0]));
    }

  // Compute third row of UT
  U[2] = Cross3(U[0], U[1]);

  // Transpose to return U
  U = Transpose3x3(U);
}

// WARNING: Results may be inaccurate if the matrix entries are several orders of magnitude above or
// below 1.
inline void
RotationVariantSvd3x3(VMatrix3x3r const& F, VMatrix3x3r& U, Vec4r& Sg, VMatrix3x3r& VT) {
  // Compute F^T * F
  VMatrix3x3r FT = Transpose3x3(F);
  VSymMatrix3x3r Gsym = SimdFullToSym(Dot3x3(FT, F));

  Vec4r eigvalues;
  AnalyticalEigendecompSym3x3(Gsym, eigvalues, &VT);

  // Compute the singular values (i.e. sqrt(eigvalues), clamp
  // them before computing sqrt in case there is rounding error
  // making them negative
  Sg = Sqrt(Max(eigvalues, Vec4r{0_r}));

  // If an inversion happened (i.e. det(𝐅) < 0), encode this
  // information by negating the smallest singular value.
  if (Det3x3(F) < 0.0) {
    Sg = Neg<0, 0, 1, 0>(Sg); // Negate flagged entries
  }

  // Compute singular vectors 𝐔=𝐅𝐕𝚺⁻¹.
  real constexpr kEpsilon = std::numeric_limits<real>::epsilon();

  // Compute first row of UT
  U[0] = DotVecMat3x3(VT[0], FT); // Faster than DotMatVec3x3
  real const u0normSqr = NormSqr<3>(U[0]);
  if (u0normSqr > kEpsilon)
    MOCHI_LIKELY {
      U[0] = Normalize(U[0], u0normSqr);
    }
  else
    MOCHI_UNLIKELY {
      U[0] = SimdBasisVector<0>();
    }

  // Compute second row of UT
  U[1] = DotVecMat3x3(VT[1], FT); // Faster than DotMatVec3x3
  real const u1normSqr = NormSqr<3>(U[1]);
  if (u1normSqr > kEpsilon)
    MOCHI_LIKELY {
      U[1] = Normalize(U[1], u1normSqr);
    }
  else
    MOCHI_UNLIKELY {
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
  MOCHI_ASSERT_VERBOSE(eps >= 0_r);

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
  MOCHI_ASSERT_VERBOSE(eps >= 0_r);
  // To improve performance and avoid accumulation of finite precision errors, project only if not
  // already positive definite. The fast-path predicate is intentionally a strict-Sylvester PD check
  // rather than a full PSD check: it requires only the three leading principal minors (cheap)
  // instead of all seven principal minors needed for a PSD test. Singular PSD matrices fall through
  // to the eigendecomposition and are projected correctly.
  if (!details::IsSpd(A)) {
    // Perform eigendecomposition.
    Vec4r L;
    VMatrix3x3r QT;
    AnalyticalEigendecompSym3x3(SimdFullToSym(A), L, &QT);

    // Perform projection.
    A = Dot3x3(Transpose3x3(QT), Dot3x3(VDiagonalMatrix<3>(Max(L, Vec4r{eps})), QT));
  }
}

inline VMatrix2x2r CholeskySym2x2(VMatrix2x2r const& A, real eps) {
  MOCHI_ASSERT_VERBOSE(eps > 0_r);

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
  Batched Eigendecomposition and SVD

  WARNING: Results may be inaccurate if the matrix entries are several orders of magnitude above
  or below 1.
*/

namespace detail {

// WARNING: `mat` must be normalized (entries within a few orders of magnitude of 1). The
// closed-form below is only accurate in that regime.
// REFERENCE: Deledalle et al., "Closed-form expressions of the eigen decomposition of 2×2 and 3×3
//   Hermitian matrices" (2017). https://hal.science/hal-01501221/document
template <int kBatchSize>
MOCHI_FORCE_INLINE void BatchedAnalyticalEigenvalsSym3x3(
    BatchSymMatrix3x3<kBatchSize> const& mat,
    BatchReal3<kBatchSize>& eigvalues) {
  using V = BatchReal<kBatchSize>;

  // Symmetric matrix layout:
  // | a d e |
  // | d b f |
  // | e f c |
  V const& a = mat[0];
  V const& b = mat[1];
  V const& c = mat[2];
  V const& d = mat[3];
  V const& e = mat[4];
  V const& f = mat[5];

  V const eps = std::numeric_limits<real>::epsilon();
  V const zero = {};
  V const diagMask = (Abs(d) <= eps) & (Abs(e) <= eps) & (Abs(f) <= eps);

  V const dSqr = d * d;
  V const eSqr = e * e;
  V const fSqr = f * f;
  V const x1 =
      Max(a * a + b * b + c * c - a * b - a * c - b * c + V{3_r} * (dSqr + eSqr + fSqr), zero);

  V const twoA_b_c = a + a - b - c;
  V const twoB_a_c = b + b - a - c;
  V const twoC_a_b = c + c - a - b;
  V const x2 = V{9_r} * (twoC_a_b * dSqr + twoB_a_c * eSqr + twoA_b_c * fSqr) -
      twoA_b_c * twoB_a_c * twoC_a_b - V{54_r} * d * e * f;

  V const absx2 = Abs(x2);
  V const x2NearZero = absx2 <= eps;

  V const one = V{1_r};
  V const safeInvX2 = one / Select(x2NearZero, one, x2);
  V const psiArg = Sqrt(Max(zero, V{4_r} * x1 * x1 * x1 - x2 * x2)) * safeInvX2;
  V psi = ATan(psiArg);
  psi = Select(x2 < zero, psi + V{kPI}, psi);
  psi = Select(x2NearZero, V{kPI * 0.5_r}, psi);

  V const trace = a + b + c;
  V const twoSqrtX1 = V{2_r} * Sqrt(x1);
  V const oneOver3 = V{1_r / 3_r};
  V const psiOver3 = psi * oneOver3;

  // Use SinCos + angle-addition: cos(θ ± π/3) = 0.5·cos(θ) ∓ (√3/2)·sin(θ)
  auto const [sinPsi3, cosPsi3] = SinCos(psiOver3);
  V const kHalf = V{0.5_r};
  V const halfCos = kHalf * cosPsi3;
  V const sqrt3HalfSin = kSqrt3Over2 * sinPsi3;

  V eigval0 = (trace - twoSqrtX1 * cosPsi3) * oneOver3;
  V eigval1 = (trace + twoSqrtX1 * (halfCos + sqrt3HalfSin)) * oneOver3;
  V eigval2 = (trace + twoSqrtX1 * (halfCos - sqrt3HalfSin)) * oneOver3;

  eigval0 = Select(diagMask, a, eigval0);
  eigval1 = Select(diagMask, b, eigval1);
  eigval2 = Select(diagMask, c, eigval2);

  // 3-element sorting network (descending)
  V const max01 = Max(eigval0, eigval1);
  V const min01 = Min(eigval0, eigval1);

  eigvalues[0] = Max(max01, eigval2);
  V const midTmp = Min(max01, eigval2);

  eigvalues[1] = Max(min01, midTmp);
  eigvalues[2] = Min(min01, midTmp);
}

// Return the largest-norm row of cofactor(A - lambda I) for the symmetric layout:
// | diag[0]    offDiag[0] offDiag[1] |
// | offDiag[0] diag[1]    offDiag[2] |
// | offDiag[1] offDiag[2] diag[2]    |
// outNormSqr returns the squared norm of the returned row.
template <int kBatchSize>
[[nodiscard]] MOCHI_FORCE_INLINE BatchReal3<kBatchSize> BatchedLargestCofactorRowSym3x3(
    BatchReal3<kBatchSize> const& diag,
    BatchReal3<kBatchSize> const& offDiag,
    BatchReal3<kBatchSize> const& offDiagSqr,
    BatchReal<kBatchSize> const& lambda,
    BatchReal<kBatchSize>& outNormSqr) {
  using V = BatchReal<kBatchSize>;

  V const& d = offDiag[0];
  V const& e = offDiag[1];
  V const& f = offDiag[2];

  V const m00 = diag[0] - lambda;
  V const m11 = diag[1] - lambda;
  V const m22 = diag[2] - lambda;

  V const cof00 = m11 * m22 - offDiagSqr[2];
  V const cof01 = e * f - d * m22;
  V const cof02 = d * f - m11 * e;
  V const cof11 = m00 * m22 - offDiagSqr[1];
  V const cof12 = d * e - m00 * f;
  V const cof22 = m00 * m11 - offDiagSqr[0];

  V const cof01Sqr = cof01 * cof01;
  V const cof02Sqr = cof02 * cof02;
  V const cof12Sqr = cof12 * cof12;
  V const norm0 = cof00 * cof00 + cof01Sqr + cof02Sqr;
  V const norm1 = cof01Sqr + cof11 * cof11 + cof12Sqr;
  V const norm2 = cof02Sqr + cof12Sqr + cof22 * cof22;

  V const use1 = norm1 > norm0;
  V const best01 = Select(use1, norm1, norm0);
  V rX = Select(use1, cof01, cof00);
  V rY = Select(use1, cof11, cof01);
  V rZ = Select(use1, cof12, cof02);

  V const use2 = norm2 > best01;
  outNormSqr = Select(use2, norm2, best01);
  rX = Select(use2, cof02, rX);
  rY = Select(use2, cof12, rY);
  rZ = Select(use2, cof22, rZ);
  return {rX, rY, rZ};
}

// WARNING: Results may be inaccurate if the matrix entries are several orders of magnitude above or
// below 1, or for tightly-clustered eigenvalues (whose eigenvectors are inherently
// ill-conditioned). Both limitations are intrinsic to analytic methods, not specific to this
// implementation.
// WARNING: Eigenvalues must be sorted in descending order.
template <int kBatchSize>
MOCHI_FORCE_INLINE void BatchedAnalyticalEigenvecsSym3x3(
    BatchSymMatrix3x3<kBatchSize> const& mat,
    BatchReal3<kBatchSize> const& eigvalues,
    BatchReal3x3<kBatchSize>& eigvecs) {
  using V = BatchReal<kBatchSize>;
  using V3 = BatchReal3<kBatchSize>;

  // Eigenvalues must be sorted in descending order (the flip rule below relies on eigvalues[1]
  // being the middle).
  MOCHI_ASSERT_VERBOSE(
      AllTrue<kBatchSize>(eigvalues[0] >= eigvalues[1]) &&
          AllTrue<kBatchSize>(eigvalues[1] >= eigvalues[2]),
      "BatchedAnalyticalEigenvecsSym3x3 requires the eigenvalues to be sorted in descending order.");

  V const eps = std::numeric_limits<real>::epsilon();
  V const min = std::numeric_limits<real>::min();
  V const zero = V{0_r};
  V3 const diag = {mat[0], mat[1], mat[2]};
  V3 const offDiag = {mat[3], mat[4], mat[5]};
  V3 const offDiagSqr = {offDiag[0] * offDiag[0], offDiag[1] * offDiag[1], offDiag[2] * offDiag[2]};
  V const one = V{1_r};

  // Flip if necessary so that first eigenvalue is the most different.
  V const diff01 = eigvalues[0] - eigvalues[1];
  V const diff12 = eigvalues[1] - eigvalues[2];
  V const flipMask = diff01 < diff12;
  V const lambda0 = Select(flipMask, eigvalues[2], eigvalues[0]);
  V const lambda2 = Select(flipMask, eigvalues[0], eigvalues[2]);

  // Compute eigvec1 from the largest cofactor row of (A - lambda0 * I).
  V lambda0CofactorNormSqr MOCHI_NO_INIT;
  V3 const r0 = BatchedLargestCofactorRowSym3x3<kBatchSize>(
      diag, offDiag, offDiagSqr, lambda0, lambda0CofactorNormSqr);

  // Normalize eigvec1. Fallback to (1,0,0) if norm is near zero.
  V const r0InvNorm = one / (Sqrt(lambda0CofactorNormSqr) + min);
  V const zeroNorm0 = lambda0CofactorNormSqr <= eps;
  V3 const e1 = {
      Select(zeroNorm0, one, r0[0] * r0InvNorm),
      Select(zeroNorm0, zero, r0[1] * r0InvNorm),
      Select(zeroNorm0, zero, r0[2] * r0InvNorm)};

  // Compute eigvec3 from the cofactor of (A - lambda2 * I), then project it onto the orthogonal
  // complement of eigvec1 and renormalize. For distinct eigenvalues, the cofactor row is already
  // orthogonal to eigvec1, so this only removes floating-point drift. For degenerate or clustered
  // lambda2, the projection (with the zeroE3 fallback below) produces a valid eigenvector
  // orthogonal to eigvec1 within the (near-)degenerate eigenspace.
  V lambda2CofactorNormSqr MOCHI_NO_INIT;
  V3 const r2 = BatchedLargestCofactorRowSym3x3<kBatchSize>(
      diag, offDiag, offDiagSqr, lambda2, lambda2CofactorNormSqr);

  V const r2InvNorm = one / (Sqrt(lambda2CofactorNormSqr) + min);
  V3 const r2Norm = r2 * r2InvNorm;
  V const e1DotR2 = Dot(e1, r2Norm);
  V3 e3 = r2Norm - e1DotR2 * e1;
  V const e3NormSqr = NormSqr(e3);
  V const e3InvNorm = one / (Sqrt(e3NormSqr) + min);
  e3 *= e3InvNorm;

  V const zeroE3 = (lambda2CofactorNormSqr <= eps) | (e3NormSqr <= eps);
  if (AnyTrue<kBatchSize>(zeroE3)) {
    V3 const o = Normalize(OrthogonalVector(e1));
    e3 = {Select(zeroE3, o[0], e3[0]), Select(zeroE3, o[1], e3[1]), Select(zeroE3, o[2], e3[2])};
  }

  V3 const e2 = Cross(e3, e1);
  // Unflip: if flipped, swap eigvec1 and eigvec3 with sign change.
  eigvecs = {
      V3{Select(flipMask, e3[0], e1[0]),
         Select(flipMask, e3[1], e1[1]),
         Select(flipMask, e3[2], e1[2])},
      e2,
      V3{Select(flipMask, -e1[0], e3[0]),
         Select(flipMask, -e1[1], e3[1]),
         Select(flipMask, -e1[2], e3[2])}};
}

// Normalize a symmetric 3x3 matrix (6 unique entries) in place by its mean absolute entry, which
// improves the numerical conditioning of the eigendecomposition. Returns the scale so eigenvalues
// can be rescaled back to the original units.
template <int kBatchSize>
[[nodiscard]] MOCHI_FORCE_INLINE BatchReal<kBatchSize> BatchedNormalizeSym3x3(
    BatchSymMatrix3x3<kBatchSize>& sym) {
  using V = BatchReal<kBatchSize>;
  V const scale =
      Max(V{1_r / 6_r} *
              (Abs(sym[0]) + Abs(sym[1]) + Abs(sym[2]) + Abs(sym[3]) + Abs(sym[4]) + Abs(sym[5])),
          V{std::numeric_limits<real>::min()});
  V const invScale = V{1_r} / scale;
  for (int i = 0; i < 6; ++i) {
    sym[i] *= invScale;
  }
  return scale;
}

} // namespace detail

template <int kBatchSize>
inline void BatchedAnalyticalEigendecompSym3x3(
    BatchSymMatrix3x3<kBatchSize> sym,
    BatchReal3<kBatchSize>& eigvalues,
    BatchReal3x3<kBatchSize>* eigvecs) {
  auto const scale = detail::BatchedNormalizeSym3x3<kBatchSize>(sym);
  detail::BatchedAnalyticalEigenvalsSym3x3<kBatchSize>(sym, eigvalues);

  if (eigvecs != nullptr) {
    detail::BatchedAnalyticalEigenvecsSym3x3<kBatchSize>(sym, eigvalues, *eigvecs);
  }

  eigvalues *= scale;
}

/**************************************************************************************************
  Batched SVD
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

// Compute U from F and VT (eigenvectors of G). Only VT rows 0 and 1 are used since U₂ = cross(U₀,
// U₁).
template <int kBatchSize>
MOCHI_FORCE_INLINE void BatchedComputeU(
    BatchReal3x3<kBatchSize> const& fm,
    BatchReal3x3<kBatchSize> const& VT,
    BatchReal3x3<kBatchSize>& U) {
  using V = BatchReal<kBatchSize>;
  using V3 = BatchReal3<kBatchSize>;
  using V3x3 = BatchReal3x3<kBatchSize>;
  V const eps = std::numeric_limits<real>::epsilon();
  V const min = std::numeric_limits<real>::min();
  V const zero = V{0_r};
  V const one = V{1_r};

  // UT row 0: u0 = F * v0
  V3 u0 = DotMatVec(fm, VT[0]);

  V const u0normSqr = NormSqr(u0);
  V const u0invNorm = one / (Sqrt(u0normSqr) + min);
  V const u0zero = u0normSqr <= eps;
  u0[0] = Select(u0zero, one, u0[0] * u0invNorm);
  u0[1] = Select(u0zero, zero, u0[1] * u0invNorm);
  u0[2] = Select(u0zero, zero, u0[2] * u0invNorm);

  // UT row 1: u1 = F * v1
  V3 u1 = DotMatVec(fm, VT[1]);

  V const u1normSqr = NormSqr(u1);
  V const u1invNorm = one / (Sqrt(u1normSqr) + min);
  V const u1zero = u1normSqr <= eps;
  u1[0] *= u1invNorm;
  u1[1] *= u1invNorm;
  u1[2] *= u1invNorm;

  if (AnyTrue<kBatchSize>(u1zero))
    MOCHI_UNLIKELY {
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

} // namespace detail

template <int kBatchSize>
inline void BatchedRotationVariantSvdVals3x3(
    BatchReal3x3<kBatchSize> const& F,
    BatchReal3<kBatchSize>& Sg) {
  BatchedRotationVariantSvdNormalEigensystem3x3<kBatchSize> normalEigensystem MOCHI_NO_INIT;
  BatchedRotationVariantSvdVals3x3<kBatchSize>(F, Sg, normalEigensystem);
}

template <int kBatchSize>
inline void BatchedRotationVariantSvdVals3x3(
    BatchReal3x3<kBatchSize> const& F,
    BatchReal3<kBatchSize>& Sg,
    BatchedRotationVariantSvdNormalEigensystem3x3<kBatchSize>& normalEigensystem) {
  using V = BatchReal<kBatchSize>;
  using V3 = BatchReal3<kBatchSize>;
  // Form normal matrix 𝐆 = 𝐅ᵀ𝐅, normalize it to improve conditioning, then eigendecompose.
  normalEigensystem.normalizedGsym = detail::BatchedComputeGsym<kBatchSize>(F);
  V const scale = detail::BatchedNormalizeSym3x3<kBatchSize>(normalEigensystem.normalizedGsym);
  detail::BatchedAnalyticalEigenvalsSym3x3<kBatchSize>(
      normalEigensystem.normalizedGsym, normalEigensystem.normalizedEigvals);
  V3 const scaledEigvals = normalEigensystem.normalizedEigvals * scale;

  // Compute singular values: σᵢ² = λᵢ. Clamp to ℝ⁺ for numerical safety.
  V const zero = {};
  Sg[0] = Sqrt(Max(scaledEigvals[0], zero));
  Sg[1] = Sqrt(Max(scaledEigvals[1], zero));
  Sg[2] = Sqrt(Max(scaledEigvals[2], zero));

  // If det(𝐅) < 0, encode inversion by negating the smallest singular value.
  Sg[2] = Select(Det(F) < zero, -Sg[2], Sg[2]);
}

// WARNING: The left singular vectors U may be inaccurate when the largest singular value is small.
template <int kBatchSize>
inline void BatchedRotationVariantSvdVecs3x3(
    BatchReal3x3<kBatchSize> const& F,
    BatchedRotationVariantSvdNormalEigensystem3x3<kBatchSize> const& normalEigensystem,
    BatchReal3x3<kBatchSize>& U,
    BatchReal3x3<kBatchSize>& VT) {
  detail::BatchedAnalyticalEigenvecsSym3x3<kBatchSize>(
      normalEigensystem.normalizedGsym, normalEigensystem.normalizedEigvals, VT);
  detail::BatchedComputeU<kBatchSize>(F, VT, U);
}

template <int kBatchSize>
inline void BatchedRotationVariantSvdValsAndVT3x3(
    BatchReal3x3<kBatchSize> const& F,
    BatchReal3<kBatchSize>& Sg,
    BatchReal3x3<kBatchSize>& VT) {
  BatchedRotationVariantSvdNormalEigensystem3x3<kBatchSize> normalEigensystem MOCHI_NO_INIT;
  BatchedRotationVariantSvdVals3x3<kBatchSize>(F, Sg, normalEigensystem);
  detail::BatchedAnalyticalEigenvecsSym3x3<kBatchSize>(
      normalEigensystem.normalizedGsym, normalEigensystem.normalizedEigvals, VT);
}

// WARNING: The left singular vectors U may be inaccurate when the largest singular value is small.
template <int kBatchSize>
inline void BatchedRotationVariantSvd3x3(
    BatchReal3x3<kBatchSize> const& F,
    BatchReal3x3<kBatchSize>& U,
    BatchReal3<kBatchSize>& Sg,
    BatchReal3x3<kBatchSize>& VT) {
  BatchedRotationVariantSvdValsAndVT3x3<kBatchSize>(F, Sg, VT);
  detail::BatchedComputeU<kBatchSize>(F, VT, U);
}

template <int kBatchSize>
inline void BatchedProjectSymPsd(BatchReal3x3<kBatchSize>& A, real eps) {
  MOCHI_ASSERT_VERBOSE(eps >= 0_r);
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
