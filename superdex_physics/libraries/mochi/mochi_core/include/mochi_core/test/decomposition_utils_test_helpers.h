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

#include <mochi_core/test/batch_helpers.h>
#include <mochi_core/utils/batch_types.h>
#include <mochi_core/utils/decomposition_utils.h>

namespace mochi::test {

template <int kBatchSize>
inline void BatchedAnalyticalEigendecompSym3x3(
    Span<Matrix3x3r const> mats,
    Span<Real3> eigvalues,
    Span<Matrix3x3r> eigvecs) {
  auto sym = LoadBatchSymMatrix3x3<kBatchSize>(mats);

  BatchReal3<kBatchSize> eigvals MOCHI_NO_INIT;
  BatchReal3x3<kBatchSize> vecs MOCHI_NO_INIT;
  BatchedAnalyticalEigendecompSym3x3<kBatchSize>(sym, eigvals, !eigvecs.empty() ? &vecs : nullptr);

  StoreBatchReal3<kBatchSize>(eigvals, eigvalues);
  if (!eigvecs.empty()) {
    StoreBatchMatrix3x3<kBatchSize>(vecs, eigvecs);
  }
}

template <int kBatchSize>
inline void BatchedRotationVariantSvdVals3x3(Span<Matrix3x3r const> F, Span<Real3> Sg) {
  auto const fm = LoadBatchMatrix3x3<kBatchSize>(F);

  BatchReal3<kBatchSize> sigma MOCHI_NO_INIT;
  BatchedRotationVariantSvdVals3x3<kBatchSize>(fm, sigma);

  StoreBatchReal3<kBatchSize>(sigma, Sg);
}

template <int kBatchSize>
inline void BatchedRotationVariantSvdValsVecs3x3(
    Span<Matrix3x3r const> F,
    Span<Matrix3x3r> U,
    Span<Real3> Sg,
    Span<Matrix3x3r> VT) {
  auto const fm = LoadBatchMatrix3x3<kBatchSize>(F);

  BatchReal3<kBatchSize> sigma MOCHI_NO_INIT;
  BatchedRotationVariantSvdNormalEigensystem3x3<kBatchSize> normalEigensystem MOCHI_NO_INIT;
  BatchedRotationVariantSvdVals3x3<kBatchSize>(fm, sigma, normalEigensystem);

  BatchReal3x3<kBatchSize> uBatch MOCHI_NO_INIT, vtBatch MOCHI_NO_INIT;
  BatchedRotationVariantSvdVecs3x3<kBatchSize>(fm, normalEigensystem, uBatch, vtBatch);

  StoreBatchReal3<kBatchSize>(sigma, Sg);
  StoreBatchMatrix3x3<kBatchSize>(uBatch, U);
  StoreBatchMatrix3x3<kBatchSize>(vtBatch, VT);
}

template <int kBatchSize>
inline void BatchedRotationVariantSvd3x3(
    Span<Matrix3x3r const> F,
    Span<Matrix3x3r> U,
    Span<Real3> Sg,
    Span<Matrix3x3r> VT) {
  auto const fm = LoadBatchMatrix3x3<kBatchSize>(F);

  BatchReal3x3<kBatchSize> uBatch MOCHI_NO_INIT, vtBatch MOCHI_NO_INIT;
  BatchReal3<kBatchSize> sigma MOCHI_NO_INIT;
  BatchedRotationVariantSvd3x3<kBatchSize>(fm, uBatch, sigma, vtBatch);

  StoreBatchReal3<kBatchSize>(sigma, Sg);
  StoreBatchMatrix3x3<kBatchSize>(uBatch, U);
  StoreBatchMatrix3x3<kBatchSize>(vtBatch, VT);
}

} // namespace mochi::test
